"""
Utility functions and classes for LoRA-FAIR federated learning.

This module contains the foundation model class and evaluation functions
for the LoRA-FAIR federated learning framework.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from models.GetModel import build_promptmodel
from peft import get_peft_model


class FoundationModel(nn.Module):
    """
    Foundation model wrapper with LoRA adaptation.
    
    This class wraps a base model with LoRA (Low-Rank Adaptation) configuration
    for parameter-efficient fine-tuning in federated learning scenarios.
    
    Args:
        layer (int): Number of layers in the model. Default: 12
        num_classes (int): Number of output classes. Default: 100
        depth_cls (int): Depth of classifier layers. Default: 0
        modeltype (str): Type of model architecture ('ViT' or 'mixer'). Default: 'ViT'
        lora_config (LoraConfig): LoRA configuration object
    """
    
    def __init__(self, layer=12, num_classes=100, depth_cls=0, modeltype='ViT', lora_config=None):
        super(FoundationModel, self).__init__()
        
        # Build the base model
        self.backbone = build_promptmodel(
            num_classes=num_classes,
            edge_size=224,
            modeltype=modeltype,
            patch_size=16,
            Prompt_Token_num=0,
            depth=layer,
            depth_cls=depth_cls
        )
        
        # Apply LoRA adaptation
        if lora_config is not None:
            self.backbone = get_peft_model(self.backbone, lora_config)
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Model output
        """
        return self.backbone(x)


def evaluation(model, test_data):
    """
    Evaluate model performance on test data.
    
    This function evaluates the model on test data and returns top-1 and top-5 accuracy.
    Supports both single test dataset and multiple test datasets.
    
    Args:
        model (nn.Module): The model to evaluate
        test_data (DataLoader or list): Test data loader(s)
        
    Returns:
        tuple: (top1_accuracy, top5_accuracy) or (list of top1, list of top5) for multiple datasets
    """
    model.eval()
    top1_accuracies, top5_accuracies = [], []
    # Handle multiple test datasets
    if isinstance(test_data, list):
        for dataset in tqdm(test_data, desc="Evaluating datasets"):
            with torch.no_grad():
                total_samples = 0
                correct_top1 = 0
                correct_top5 = 0
                
                for test_images, test_labels in dataset:
                    test_labels = test_labels.cuda()
                    outputs = model(test_images.cuda())
                    
                    # Get top-5 predictions
                    _, top5_predictions = torch.topk(outputs, 5, dim=-1)
                    
                    total_samples += test_labels.size(0)
                    test_labels = test_labels.view(-1, 1)
                    
                    # Calculate top-1 and top-5 accuracy
                    correct_top1 += (test_labels == top5_predictions[:, 0:1]).sum().item()
                    correct_top5 += (test_labels == top5_predictions).sum().item()
            
            # Convert to percentage
            top1_accuracies.append(100 * correct_top1 / total_samples)
            top5_accuracies.append(100 * correct_top5 / total_samples)
        
        return top1_accuracies, top5_accuracies
    
    # Handle single test dataset
    else:
        with torch.no_grad():
            total_samples = 0
            correct_top1 = 0
            correct_top5 = 0
            
            for test_images, test_labels in test_data:
                test_labels = test_labels.cuda()
                outputs = model(test_images.cuda())
                
                # Get top-5 predictions
                _, top5_predictions = torch.topk(outputs, 5, dim=-1)
                
                total_samples += test_labels.size(0)
                test_labels = test_labels.view(-1, 1)
                
                # Calculate top-1 and top-5 accuracy
                correct_top1 += (test_labels == top5_predictions[:, 0:1]).sum().item()
                correct_top5 += (test_labels == top5_predictions).sum().item()
        
        return 100 * correct_top1 / total_samples, 100 * correct_top5 / total_samples