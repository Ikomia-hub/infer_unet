# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
import os.path
from ikomia import core, dataprocess
import copy
from infer_unet.unet import UNet
import torch
from infer_unet.predict.utils_prediction import predict_mask, unet_carvana


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferUnetParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_weight_file = ""
        self.input_size = 128

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_weight_file = param_map["model_weight_file"]
        self.input_size = int(param_map["input_size"])
        pass

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "model_weight_file": str(self.model_weight_file),
            "input_size": str(self.input_size)
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferUnet(dataprocess.CSemanticSegmentationTask):

    def __init__(self, name, param):
        dataprocess.CSemanticSegmentationTask.__init__(self, name)

        self.net = None
        self.classes = None

        # Create parameters class
        if param is None:
            self.set_param_object(InferUnetParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # we use seed to keep the same color for our masks + boxes + labels (same random each time)

        # Get input :
        img_input = self.get_input(0)
        input_image = img_input.get_image()

        # Get parameters :
        param = self.get_param_object()

        # load model file after training unet model or use the Carnava pretrained model to test the model
        path = param.model_weight_file
        # load model
        if self.net is None or param.update:
            if os.path.isfile(path):
                # load model dict
                model_dict = torch.load(path)
                # load class names from model dict
                classes_dict = model_dict['class_names']
                self.classes = list(classes_dict.values())

                n_class = len(self.classes)
                # load state_dict
                state_dict = model_dict['state_dict']
                self.net = UNet(input_image.shape[2], n_class)
                self.net.load_state_dict(state_dict)

            # else use the carnava pretrained model
            else:
                self.classes = ['background', 'car']
                try:
                    self.net = unet_carvana(pretrained=True, scale=0.5)
                except:
                    self.net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=0.5)

            self.set_names(self.classes)
            param.update = False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(device=device)
        self.net.eval()

        mask = predict_mask(net=self.net,
                            full_img=input_image,
                            size=param.input_size,
                            device=device)
        mask = mask.astype('uint8')

        # Set the mask of the semantic segmentation output
        self.set_mask(mask)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferUnetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_unet"
        self.info.short_description = "Multi-class semantic segmentation using Unet, " \
                                      "the default model was trained on Kaggle's Carvana Images dataset"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.icon_path = "icon/unet.jpg"
        self.info.version = "1.1.0"
        self.info.license = "GPL-3.0 license"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "Olaf Ronneberger, Philipp Fischer, Thomas Brox"
        self.info.article = "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        self.info.year = 2015
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_unet"
        self.info.original_repository = "https://github.com/milesial/Pytorch-UNet"
        # Keywords used for search
        self.info.keywords = "semantic segmentation, unet, multi-class segmentation"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "SEMANTIC_SEGMENTATION"

    def create(self, param=None):
        # Create process object
        return InferUnet(self.info.name, param)
