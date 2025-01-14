from pickle import LONG_BINGET
import torch
import torch.nn as nn
import sys

import subprocess


matplotlib_required_version = '3.7.1' 
subprocess.check_call([sys.executable, "-m", "pip", "install", f"matplotlib=={matplotlib_required_version}"])

sys.path.append('/content/drive/MyDrive/Neural-Highlighting-of-Affordance-Regions/PointCLIP/Dassl3D')
from dassl.engine import TRAINER_REGISTRY, TrainerX

from clip import clip
#from trainers.mv_utils_zs import PCViews
from mv_utils_zs import PCViews
from mv_utils_zs import point_cloud_render_to_tensor
import matplotlib.pyplot as plt

MODEL_BACKBONE='ViT-B/16'
MODEL_BACKBONE_CHANNEL = 512
NUM_VIEWS = 6
TEXTUAL_PROMPT = 'a scatter plot of a point cloud representing a gray dog with highlighted shoes'

def get_clip(clipmodel, device='cuda'):
    model, _ = clip.load(clipmodel, device=device)
    return model
    
def load_clip_to_cpu(backbone_name='ViT-B/16'):             #oppure ViT-B/32
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, root='./clip_models')
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model

class Textual_Encoder(nn.Module):

    def __init__(self, textual_prompt, clip_model_name=MODEL_BACKBONE):        #clip_model = clip_model_name
        super().__init__()
        self.clip_model = get_clip(clip_model_name)
        self.dtype = self.clip_model.dtype
        self.textual_prompt = textual_prompt
    
    def forward(self):
        prompts = self.textual_prompt
        #prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = clip.tokenize([prompts])

        prompts = prompts.cuda()
        text_feat = self.clip_model.encode_text(prompts).repeat(1, NUM_VIEWS)       #6 è num_views di depth maps
        #print(text_feat.shape)
        return text_feat
        '''
        -----This is our code----
        with torch.no_grad():
        text_input = clip.tokenize([prompt]).to(device)
        encoded_text = clip_model.encode_text(text_input)
        encoded_text = encoded_text / encoded_text.norm(dim=1, keepdim=True)
        '''


@TRAINER_REGISTRY.register()
class PointCLIP_ZS(TrainerX):

    #this is an overridden function, the original is without parameters,
    #so to make it works, the textual prompt is temporarily hardcoded. 
    #TODO: check if it is better to add an __init__ here to override the original build_model()
    
    def build_model(self):

        print(f'Loading CLIP (backbone: {MODEL_BACKBONE})')
        clip_model = load_clip_to_cpu(MODEL_BACKBONE)
        clip_model.cuda()

        # Encoders from CLIP
        self.visual_encoder = clip_model.encode_image
        self.textual_encoder = Textual_Encoder(TEXTUAL_PROMPT)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.channel = MODEL_BACKBONE_CHANNEL
    
        # Multi-view projection
        self.num_views = NUM_VIEWS
        pc_views = PCViews()
        self.get_img = point_cloud_render_to_tensor
        #self.get_img = pc_views.get_img_with_color
  
        # Store features for post-process view-weight search
        self.feat_store = []
        self.label_store = []
        
    '''
    def mv_proj(self, pc):
        img = self.get_img(pc).cuda()
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)
        img = torch.nn.functional.upsample(img, size=(224, 224), mode='bilinear', align_corners=True)
        return img
    '''
    
    
    def mv_proj(self, pc):
        img = self.get_img(pc).cuda()
        #img = self.get_img(pc).cuda()
        #img = img.unsqueeze(1).repeat(1, 3, 1, 1) #this simulates the RGB colors, instead we have only to permute the tensor (we already have color info)
        
        
        print("img shape zeroshot: ", img.shape)
        #img = img.squeeze(1)
        
        #img = img.permute(0, 3, 1, 2)
        #img = torch.nn.functional.upsample(img, size=(224, 224), mode='bilinear', align_corners=True)
        return img

    def model_inference(self, pc, label=None):

        # Project to multi-view depth maps
        images = self.mv_proj(pc).type(self.dtype)

        images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # Shape: [6, 224, 224, 3]

        # Create a grid to display the 6 images
        
        fig, axes = plt.subplots(1, 6, figsize=(15, 5))
        for i, ax in enumerate(axes):
            ax.imshow(images_np[i])
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        
        #DEBUG
        #print(images.shape) #OK [6,3,224,224]
        
        with torch.no_grad():
            # Image features
            image_feat = self.visual_encoder(images)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True) 
            image_feat = image_feat.reshape(-1, self.num_views * self.channel)
            
            #DEBUG
            #print(image_feat.shape) OK [1, n_views*channel]

            # Store for zero-shot
            self.feat_store.append(image_feat)
            self.label_store.append(label)

            # Text features
            text_feat = self.textual_encoder()
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  
            
            #DEBUG
            #print(text_feat.shape)

            # Classification logits
            logit_scale = self.logit_scale.exp()
            #print("logit scale shape:", logit_scale.shape)
            #print(logit_scale)

            test = image_feat @ text_feat.t()
            #logits = logit_scale * image_feat @ text_feat.t() * 1.0
            logits = image_feat @ text_feat.t() * 1.0
            
        return logits.item()

    def model_inference_img(self, pc, colors):

        # Project to multi-view depth maps
        images = self.mv_proj(pc, colors).type(self.dtype)

        with torch.no_grad():
            # Image features
            image_feat = self.visual_encoder(images)
        
        return image_feat
