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
from infer_unet.predict.utils_prediction import predict_mask, mask_to_image, unet_carvana
from PIL import Image
from datetime import datetime
import numpy as np
import random
# Your imports below


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferUnetParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.modelFile = ""
        self.img_size = 128

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.modelFile = param_map["modelFile"]
        self.img_size = int(param_map["img_size"])
        pass

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["modelFile"] = str(self.modelFile)
        param_map["img_size"] = str(self.img_size)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferUnet(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # add output
        self.addOutput(dataprocess.CSemanticSegIO())
        self.colors = None

        # Create parameters class
        if param is None:
            self.setParam(InferUnetParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        self.colors = None

        # Get input :
        img_input = self.getInput(0)
        input_image = img_input.getImage()

        # Get parameters :
        param = self.getParam()

        # load model file after training unet model or use the Carnava pretrained model to test the model
        path = param.modelFile
        if os.path.isfile(path):
            # load model dict
            model_dict = torch.load(path)
            # load class names from model dict
            classes_dict = model_dict['class_names']
            classes = list(classes_dict.values())

            n_class = len(classes)
            # load state_dict
            state_dict = model_dict['state_dict']
            net = UNet(input_image.shape[2], n_class)
            net.load_state_dict(state_dict)

        # else use the carnava pretrained model
        else:
            classes = ['background', 'car']
            n_class = 2
            self.colors = [[0, 0, 0], [255, 0, 0]]
            net = unet_carvana(pretrained=False, scale=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device=device)
        net.eval()

        mask = predict_mask(net=net,
                            full_img=input_image,
                            size=param.img_size,
                            device=device)
        mask = mask.astype('uint8')

        # Get output :
        output = self.getOutput(1)
        # Set the mask of the semantic segmentation output
        output.setMask(mask)

        # create color map
        if self.colors is None:
            self.create_color_map(n_class)

        print('classes', classes)
        print('colors', self.colors)
        output.setClassNames(classes, self.colors)

        # Apply color map on labelled image
        self.setOutputColorMap(0, 1, self.colors)
        self.forwardInputImage(0, 0)

        # Get output :
        # output = self.getOutput(0)
        # Set image of output (numpy array):
        # output.setImage(mask_image)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def create_color_map(self, num_classes):
            self.colors = []
            for i in range(num_classes):
                self.colors.append([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)])
            return self.colors
# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferUnetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_unet"
        self.info.shortDescription = "multi-class semantic segmentation using Unet"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "Olaf Ronneberger, Philipp Fischer, Thomas Brox"
        self.info.article = "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        self.info.year = 2015
        # Code source repository
        self.info.repository = "https://github.com/milesial/Pytorch-UNet"
        # Keywords used for search
        self.info.keywords = "semantic segmentation, unet, multi-class segmentation"

    def create(self, param=None):
        # Create process object
        return InferUnet(self.info.name, param)
