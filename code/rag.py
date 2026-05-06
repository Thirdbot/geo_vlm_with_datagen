from collections import defaultdict

import numpy as np
import pymupdf as fitz
import segyio
from llama_index.core import Settings, StorageContext
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import ImageNode, TextNode
from llama_index.embeddings.clip import ClipEmbedding
from pathlib import Path
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from PIL import Image
from tqdm import tqdm


class RAG:
    def __init__(self,source_path,storage_path):
        self.root = Path(__file__).parent.parent.absolute()
        self.source_path = Path(source_path)
        self.client = QdrantClient(path=Path(storage_path).as_posix())
        self.text_collection = "clip_text_store_pdf_seismic"
        self.image_collection = "clip_image_store_pdf_seismic"
        # set vector store
        self.text_vector = QdrantVectorStore(client=self.client, collection_name=self.text_collection)
        self.image_vector = QdrantVectorStore(client=self.client, collection_name=self.image_collection)
        # use CLIP for both text and images so they share the same embedding space
        self.embed_model = ClipEmbedding()
        self.image_embed_model = ClipEmbedding()
        # placeholder llm
        self.llm = MockLLM()
        Settings.llm = self.llm
        # chunking texts
        self.text_parser = SentenceSplitter(chunk_size=64, chunk_overlap=10, include_metadata=False)


        #fault path
        self.fault_sticks_path = self.source_path.joinpath("seismic","fault_sticks","Fault_Sticks","data","fault_Sticks_GN1101_2012")
        self.seismic_3d_surveys_path = self.source_path.joinpath("seismic","seismic_3d_surveys","Seismic_3D_Surveys","data","GN1101_Scaled(Realized)")

        # context from text vector
        storage_context = StorageContext.from_defaults(vector_store=self.text_vector, image_store=self.image_vector)


        if not self.client.collection_exists(self.text_collection) or not self.client.collection_exists(self.image_collection):

            # combine multiple source
            nodes = []
            nodes.extend(self.extracts_from_pdfs())
            nodes.extend(self.extract_fault())

            # feed data to multimodal embedding store
            self.index = MultiModalVectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                image_vector_store=self.image_vector,
                embed_model=self.embed_model,
                image_embed_model=self.image_embed_model,
                show_progress=True
            )
        else:
            self.index = MultiModalVectorStoreIndex.from_vector_store(
                vector_store=self.text_vector,
                image_vector_store=self.image_vector,
                embed_model=self.embed_model,
                image_embed_model=self.image_embed_model,
            )
    def read_fault_stick(self):
        # section_type,inline,crossline,x,y,z,fault_name,stick_id
        with open(self.fault_sticks_path,'r',encoding="latin-1") as read_fault_meta:
            # find each column
            for line in read_fault_meta:
                columns_data = line.split()
                if len(columns_data) == 8:
                    yield columns_data

    @staticmethod
    def _nearest_index(values, value):
        return int(np.abs(values - value).argmin())

    @staticmethod
    def _draw_line(mask, start, end, thickness=2):
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
        radius = max(0, thickness // 2)

        for x, y in zip(
            np.linspace(x0, x1, steps).round().astype(int),
            np.linspace(y0, y1, steps).round().astype(int),
        ):
            y_min = max(0, y - radius)
            y_max = min(mask.shape[0], y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(mask.shape[1], x + radius + 1)
            mask[y_min:y_max, x_min:x_max] = 255

    def extract_fault(self):
        groups = defaultdict(list)
        seismic_image_dir = self.root / "storage" / "seismic_images"
        seismic_image_dir.mkdir(exist_ok=True, parents=True)
        nodes = []


        # use fault metadata
        for section_type,inline,crossline,x,y,z,fault_name,stick_id in tqdm(
            self.read_fault_stick(),
            desc="Grouping fault sticks",
        ):
            # convert type
            inline = int(inline)
            crossline = int(crossline)
            z = float(z)
            stick_id = int(stick_id)

            # group from fault_stick and extract data
            groups[(inline, fault_name, stick_id)].append({
                "crossline": crossline,
                "z": z,
                "x": float(x),
                "y": float(y),
            })

        with segyio.open(self.seismic_3d_surveys_path.as_posix(),'r') as seismic:
            xlines = seismic.xlines
            seismic_samples = seismic.samples

            for (inline,fault_name,stick_id),points in tqdm(
                groups.items(),
                desc="Building seismic fault images",
                total=len(groups),
            ):

                if inline not in seismic.ilines:
                    continue

                # segyio returns inline slices as (xlines, samples).
                # Transpose to image layout: (height=samples, width=xlines).
                slice_2d = seismic.iline[inline].T # slice for original image
                mask = np.zeros(slice_2d.shape, dtype=np.uint8) # mask for segmentation train

                fault_pixels = []
                for point in points:
                    x_idx = self._nearest_index(xlines, point["crossline"])
                    y_idx = self._nearest_index(seismic_samples, point["z"])

                    if 0 <= y_idx < mask.shape[0] and 0 <= x_idx < mask.shape[1]:
                        fault_pixels.append((x_idx, y_idx)) # create list of fault-pixel

                for start, end in zip(fault_pixels, fault_pixels[1:]):
                    self._draw_line(mask, start, end)

                if len(fault_pixels) == 1:
                    x_idx, y_idx = fault_pixels[0]
                    mask[y_idx, x_idx] = 255 # draw it white



                # samples.append((slice_2d, mask))
                safe_fault_name = "".join(
                    char if char.isalnum() or char in {"-", "_"} else "_"
                    for char in fault_name
                )
                stem = f"inline_{inline}_{safe_fault_name}_stick_{stick_id}"
                seismic_path = seismic_image_dir / f"{stem}_seismic.jpeg"
                seismic_norm = slice_2d
                seismic_norm = seismic_norm - seismic_norm.min()
                seismic_norm = seismic_norm / seismic_norm.max()
                seismic_img = (seismic_norm * 255).astype("uint8")

                fault_path = seismic_image_dir / f"{stem}_fault_mask.png"
                fault_img = mask
                Image.fromarray(seismic_img).save(seismic_path)
                Image.fromarray(fault_img).save(fault_path)

                overlay_path = seismic_image_dir / f"{stem}_overlay.png"
                seismic_rgb = np.stack([seismic_img, seismic_img, seismic_img], axis=-1) # overlay for result image
                seismic_rgb[mask > 0] = [255, 0, 0]
                Image.fromarray(seismic_rgb).save(overlay_path)

                metadata = {
                    "source": seismic_path.as_posix(),
                    "page": None, # make it so fault can't rank
                    "fault_mask_path": fault_path.as_posix(), # put reference to fault cause similarity is not enough for it
                    "fault_overlay_path": overlay_path.as_posix(), # pu reference to overlay cause similarity is not enough for it
                }
                # pack seismic to pass in RAG
                nodes.append(
                    ImageNode(
                        text=f"seismic {seismic_path.name} image",
                        image_path=seismic_path.as_posix(),
                        metadata={**metadata, "type": "image"},
                        excluded_embed_metadata_keys=[
                            "source",
                            "page",
                            "type",
                            "fault_mask_path",
                            "fault_overlay_path",
                        ]
                    )
                )

                metadata = {
                    "source": fault_path.as_posix(),
                    "page": None # make it so fault can't rank
                }
                # pack fault to pass in RAG
                nodes.append(
                    ImageNode(
                        text=f"fault mask {fault_path.name} image",
                        image_path=fault_path.as_posix(),
                        metadata={**metadata, "type": "image_target"},
                        excluded_embed_metadata_keys=["source", "page", "type"]
                    )
                )


                metadata = {
                    "source": overlay_path.as_posix(),
                    "page": None # make it so fault can't rank
                }
                # pack overlay to pass in RAG
                nodes.append(
                    ImageNode(
                        text=f"overlay {overlay_path.name} image",
                        image_path=overlay_path.as_posix(),
                        metadata={**metadata, "type": "image"},
                        excluded_embed_metadata_keys=["source", "page", "type"]
                    )
                )

        return nodes

    def extracts_from_pdfs(self):
        nodes = []
        image_dir = self.root / "storage" / "pdf_images"
        image_dir.mkdir(parents=True, exist_ok=True)

        pdf_paths = list(Path(self.source_path).rglob('*.pdf'))
        for path in tqdm(pdf_paths, desc="Extracting PDF nodes"):

            doc = fitz.open(path.as_posix())

            for page_number, page in enumerate(doc.pages()):
                text = page.get_text()
                metadata = {
                    "source": path.as_posix(),
                    "page": page_number + 1,
                }

                if text.strip():
                    for chunk in self.text_parser.split_text(text):
                        nodes.append(
                            TextNode(
                                text=chunk,
                                metadata={**metadata, "type": "text"},
                                excluded_embed_metadata_keys=["source", "page", "type"], # not embedded leaving it for reference only
                            )
                        )

                for image_number, image_info in enumerate(page.get_images(full=True), start=1):
                    xref = image_info[0]
                    image = doc.extract_image(xref)
                    image_path = image_dir / f"{path.stem}-page-{page_number + 1}-image-{image_number}.{image['ext']}"
                    image_path.write_bytes(image["image"])
                    nodes.append(
                        ImageNode(
                            text=f"PDF embedded image: {path.name} page {page_number + 1} image {image_number}",
                            image_path=image_path.as_posix(),
                            metadata={**metadata, "type": "image", "image": image_number},
                            excluded_embed_metadata_keys=["source", "page", "type", "image"],
                        )
                    )

            doc.close()

        return nodes

    def query(self, query_im,query_q):
        query_engine = self.index.as_query_engine(llm=self.llm)
        response_r = query_engine.image_query(query_im,query_q)
        return response_r

    def retrieve(self, query,text_top_k,image_top_k):
        retriever = self.index.as_retriever(
            similarity_top_k=text_top_k,
            image_similarity_top_k=image_top_k,
        )
        retrieve_r = retriever.retrieve(query)
        results = []

        for item in retrieve_r:
            node = item.node
            image_path = getattr(node, "image_path", None)
            results.append(
                {
                    "type": node.metadata.get("type"),
                    "score": item.score,
                    "node_id": node.node_id,
                    "content": node.get_content(),
                    "source": node.metadata.get("source"),
                    "page": node.metadata.get("page"),
                    "metadata": node.metadata,
                    "image_path": image_path,
                }
            )

        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def close(self):
        self.client.close()

# if __name__ == "__main__":
#     root = Path(__file__).parent.parent.absolute()
#     storage_path = root / "storage"
#     storage_path.mkdir(parents=True, exist_ok=True)
#     source_path = root.joinpath("source")
#     test_img_path = root.joinpath("test")
#     r = RAG(source_path=source_path.as_posix(),storage_path=storage_path.as_posix())
#     # try:
#     #     answer = r.query(
#     #     test_img_path.joinpath("img.png").as_posix()
#     #     ,"what is in pdf?")
#     #     print(answer)
#     #
#     #     retrieve = r.retrieve(
#     #         "fault interpretation",3,3)
#     #     print(retrieve)
#     # finally:
#     #     r.close()
#     r.extract_fault()
