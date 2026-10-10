from stable_datasets.schema import ClassLabel, DatasetInfo, DatasetSource, DownloadInfo, Features, Version
from stable_datasets.schema import Image as ImageFeature
from stable_datasets.utils import BaseDatasetBuilder


class DeepWeeds(BaseDatasetBuilder):
    """DeepWeeds dataset for classification of eight classes of weeds as well as a negative class in the Australian rangelands."""

    VERSION = Version("1.0.0")

    # Single source-of-truth for dataset provenance + download locations.
    SOURCE = DatasetSource(
        homepage="https://github.com/AlexOlsen/DeepWeeds",
        assets={
            "images": DownloadInfo(
                url="https://drive.usercontent.google.com/download?id=1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj&export=download&confirm=t",
                filename="deepweeds_images.zip",
            ),
            "labels": DownloadInfo(
                url="https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/da084f62bcd1a2b0afeb8b81f1a27be3186391a1/labels/labels.csv",
                filename="deepweeds_labels.csv",
            ),
        },
        citation="""@article{DeepWeeds2019,
                        author = {Alex Olsen and
                            Dmitry A. Konovalov and
                            Bronson Philippa and
                            Peter Ridd and
                            Jake C. Wood and
                            Jamie Johns and
                            Wesley Banks and
                            Benjamin Girgenti and
                            Owen Kenny,
                        title = {{DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning}},
                        journal = {Scientific Reports},
                        year = 2019,
                        number = 2058,
                        month = 2,
                        volume = 9,
                        issue = 1,
                        day = 14,
                        url = "https://doi.org/10.1038/s41598-018-38343-3",
                        doi = "10.1038/s41598-018-38343-3"
                        }
                        """,
    )

    def _info(self):
        return DatasetInfo(
            description="""
                            DeepWeeds is an image classification dataset containing 17,509 in-situ images
                            collected across several locations in Queensland, Australia. The dataset includes
                            eight invasive weed species—Chinee apple, Lantana, Parkinsonia, Parthenium,
                            Prickly acacia, Rubber vine, Siam weed, and Snake weed as well as a negative
                            class containing images without the target weed species. Images were captured
                            under natural field conditions with surrounding vegetation, providing a
                            realistic benchmark for weed species recognition and agricultural computer vision.
                        """,
            features=Features(
                {
                    "image": ImageFeature(),
                    "label": ClassLabel(
                        names=[
                            "Chinee_apple",
                            "Lantana",
                            "Parkinsonia",
                            "Parthenium",
                            "Prickly_acacia",
                            "Rubber_vine",
                            "Siam_weed",
                            "Snake_weed",
                            "Negative",
                        ]
                    ),
                }
            ),
            supervised_keys=("image", "label"),
            homepage=self.SOURCE["homepage"],
            license="CC BY 4.0",
            citation=self.SOURCE["citation"],
        )
