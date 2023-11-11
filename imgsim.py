import torch
import os
import math

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from kmeans_pytorch import kmeans
from PIL import Image


class Img2Vec:
    def __init__(self, model_name, weights="DEFAULT"):
        self.embed_dict = {
            "resnet50": self.obtain_children,
            "vgg19": self.obtain_classifier,
            "efficientnet_b0": self.obtain_classifier,
        }

        self.architecture = self.validate_model(model_name)
        self.weights = weights
        self.transform = self.assign_transform(weights)
        self.device = self.set_device()
        self.model = self.initiate_model()
        self.embed = self.assign_layer()
        self.dataset = {}
        self.image_clusters = {}
        self.cluster_centers = {}

    def validate_model(self, model_name):
        if model_name not in self.embed_dict.keys():
            raise ValueError(f"The model {model_name} is not supported")
        else:
            return model_name

    def assign_transform(self, weights):
        weights_dict = {
            "resnet50": models.ResNet50_Weights,
            "vgg19": models.VGG19_Weights,
            "efficientnet_b0": models.EfficientNet_B0_Weights,
        }

        try:
            w = weights_dict[self.architecture]
            weights = getattr(w, weights)
            preprocess = weights.transforms()
        except Exception:
            preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        return preprocess

    def set_device(self):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"

        return device

    def initiate_model(self):
        m = getattr(
            models, self.architecture
        )  
        model = m(weights=self.weights) 
        model.to(self.device)

        return model.eval()

    def assign_layer(self):
        model_embed = self.embed_dict[self.architecture]()

        return model_embed

    def obtain_children(self):
        model_embed = nn.Sequential(*list(self.model.children())[:-1])

        return model_embed

    def obtain_classifier(self):
        self.model.classifier = self.model.classifier[:-1]

        return self.model

    def directory_to_list(self, dir):
        ext = (".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG")

        d = os.listdir(dir)
        source_list = [os.path.join(dir, f) for f in d if os.path.splitext(f)[1] in ext]
        return source_list

    def validate_source(self, source):
        # convert source format into standard list of file paths
        if isinstance(source, list):
            source_list = [f for f in source if os.path.isfile(f)]
        elif os.path.isdir(source):
            source_list = self.directory_to_list(source)
        elif os.path.isfile(source):
            source_list = [source]
        else:
            raise ValueError('"source" expected as file, list or directory.')

        return source_list

    def embed_image(self, img):
        # load and preprocess image
        img = img.convert('RGB')
        img_trans = self.transform(img)

        # store computational graph on GPU if available
        if self.device == "cuda:0":
            img_trans = img_trans.cuda()

        img_trans = img_trans.unsqueeze(0)

        return self.embed(img_trans)

    def embed_dataset(self, source):
        # convert source to appropriate format
        self.files = self.validate_source(source)

        for file in self.files:
            vector = self.embed_image(file)
            self.dataset[str(file)] = vector

        return

    def similar_images(self, target_file, threshold=0.5, n=1):
        target_vector = self.embed_image(target_file)

        if 'tensors.pt' in os.listdir():
            self.load_dataset("tensors.pt")

        # initiate computation of consine similarity
        cosine = nn.CosineSimilarity(dim=1)

        # iteratively store similarity of stored images to target image
        sim_dict = {}
        print(sim_dict)
        for k, v in self.dataset.items():
            sim = cosine(v, target_vector)[0].item()
            sim_dict[k] = sim

        # sort based on decreasing similarity
        items = sim_dict.items()
        sim_dict = {k: v for k, v in sorted(items, key=lambda i: i[1], reverse=True)}

        # cut to defined top n similar images
        sim_list = list(sim_dict.items())[: int(n)]
        print(sim_list)
        if n is not None:
            sim_dict = dict(sim_list)

        if len(sim_list) == 0 or sim_list[0][1] < threshold:
            self.dataset[str(target_file)] = target_vector
            print(self.dataset)
            self.save_dataset("")
            if len(sim_list) == 0:
                return 0

        return sim_list[0][1]

    def output_images(self, similar, target):
        self.display_img(target, "original")

        for k, v in similar.items():
            self.display_img(k, "similarity:" + str(v))

        return

    def display_img(self, path, title):
        plt.imshow(path)
        plt.axis("off")
        plt.title(title)
        plt.show()

        return

    def save_dataset(self, path):
        data = {"model": self.architecture, "embeddings": self.dataset}

        torch.save(
            data, os.path.join(path, "tensors.pt")
        )  

        return

    def load_dataset(self, source):
        data = torch.load(source)

        # assess that embedding nn matches currently initiated nn
        if data["model"] == self.architecture:
            self.dataset = data["embeddings"]
        else:
            raise AttributeError(
                f'NN architecture "{self.architecture}" does not match the '
                + f'"{data["model"]}" model used to generate saved embeddings.'
                + " Re-initiate Img2Vec with correct architecture and reload."
            )

        return

    def plot_list(self, img_list, cluster_num):
        fig, axes = plt.subplots(math.ceil(len(img_list) / 2), 2)
        fig.suptitle(f"Cluster: {str(cluster_num)}")
        [ax.axis("off") for ax in axes.ravel()]

        for img, ax in zip(img_list, axes.ravel()):
            ax.imshow(Image.open(img))

        fig.tight_layout()

        return

    def display_clusters(self):
        for num in self.cluster_centers.keys():
            img_list = [k for k, v in self.image_clusters.items() if v == num]
            self.plot_list(img_list, num)

        return

    def cluster_dataset(self, nclusters, dist="euclidean", display=False):
        vecs = torch.stack(list(self.dataset.values())).squeeze()
        imgs = list(self.dataset.keys())
        np.random.seed(100)

        cluster_ids_x, cluster_centers = kmeans(
            X=vecs, num_clusters=nclusters, distance=dist, device=self.device
        )

        self.image_clusters = dict(zip(imgs, cluster_ids_x.tolist()))

        cluster_num = list(range(0, len(cluster_centers)))
        self.cluster_centers = dict(zip(cluster_num, cluster_centers.tolist()))

        if display:
            self.display_clusters()

        return
