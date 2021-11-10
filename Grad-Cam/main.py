import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
import ssl
from metaflow import FlowSpec, step, conda, conda_base
from utils import *


@conda_base(python='3.8', libraries={'matplotlib': '3.3.4', 'numpy': '1.20.1', 'tensorflow': '2.7.0'})
class GradCamVisualization(FlowSpec):
    ssl._create_default_https_context = ssl._create_unverified_context


    @step
    def start(self):
        self.img_size = (299, 299)
        self.last_conv_layer_name = "block14_sepconv2_act"
        self.preprocess_input = keras.applications.xception.preprocess_input
        # The local path to our target image
        self.img_path = keras.utils.get_file(
            "african_elephant.jpg", "https://i.imgur.com/Bvro0YD.png"
        )
        self.next(self.generate_heatmap)

    
    @step
    def generate_heatmap(self):
        import os
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.model_builder = keras.applications.xception.Xception
        self.decode_predictions = keras.applications.xception.decode_predictions

        img_array = keras.applications.xception.preprocess_input(get_img_array(self.img_path, size=self.img_size))

        # Make model
        model = self.model_builder(weights="imagenet")

        # Remove last layer's softmax
        model.layers[-1].activation = None

        # Print what the top predicted class is
        preds = model.predict(img_array)
        print("Predicted:", self.decode_predictions(preds, top=1)[0])

        # This code fails with segmentation fault
        heatmap = make_gradcam_heatmap(img_array, model, self.last_conv_layer_name)
        matplotlib.use('tkagg')
        plt.matshow(heatmap)
        plt.savefig("heatmap.jpg")
        plt.show()
        save_and_display_gradcam(self.img_path, heatmap)
        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == "__main__":
    GradCamVisualization()