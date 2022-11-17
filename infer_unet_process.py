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
from infer_unet.predict.utils_prediction import predict_mask, mask_to_image
from PIL import Image
from datetime import datetime
import numpy as np
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
        self.img_scale = 0.5
        self.num_channels = 3
        self.num_classes = 4
        self.outputFolder= ""

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.modelFile = param_map["modelFile"]
        self.img_scale = param_map["img_scale"]
        self.num_channels = param_map["num_channels"]
        self.num_classes = param_map["num_classes"]
        self.outputFolder = param_map["outputFolder"]
        pass

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["modelFile"] = self.modelFile
        param_map["img_scale"] = self.img_scale
        param_map["num_channels"] = self.num_channels
        param_map["num_classes"] = self.num_classes
        param_map["outputFolder"] = self.outputFolder
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferUnet(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)

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

        # Get input :
        img_input = self.getInput(0)
        input_image = img_input.getImage()

        # Get parameters :
        param = self.getParam()

        # load model file after training unet model or use the Carnava pretrained model to test the model
        path = param.modelFile
        if os.path.isfile(path):
            net = UNet(param.num_channels, param.num_classes)
            net.load_state_dict(torch.load(path))
        # else use the carnava pretrained model
        else:
            net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=True, scale=param.img_scale)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device=device)
        net.eval()

        # segment image
        # mask prédit par le modèle (probabilités de chaque classe)
        img = Image.fromarray(input_image)
        mask = predict_mask(net=net,
                            full_img=img,
                            scale_factor=param.img_scale,
                            device=device)

        # convert prediction to image
        mask_image = mask_to_image(mask)
        Pil_img = Image.fromarray(mask_image.astype(np.uint8))
        # save segmented image
        # current datetime is used as folder name
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        out_img_path = os.path.join(param.outputFolder, str_datetime+'.png')
        Pil_img.save(out_img_path)

        # Get output :
        output = self.getOutput(0)

        # Set image of output (numpy array):
        output.setImage(mask_image)


        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferUnetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_unet"
        self.info.shortDescription = "your short description"
        self.info.description = "your description"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "algorithm author"
        self.info.article = "title of associated research article"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "MIT License"
        # URL of documentation
        self.info.documentationLink = ""
        # Code source repository
        self.info.repository = ""
        # Keywords used for search
        self.info.keywords = "your,keywords,here"

    def create(self, param=None):
        # Create process object
        return InferUnet(self.info.name, param)
