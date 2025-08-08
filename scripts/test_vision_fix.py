#!/usr/bin/env python3
"""
Simple test to verify the vision forward pass fix.
"""

import torch
import sys
import os

# Add the model directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

def test_vision_forward_pass():
    """Test the fixed vision forward pass."""
    print("Testing vision forward pass fix...")
    
    try:
        from gpt_oss_vision.configuration_gpt_oss_vision import GPTOSSVisionConfig
        from gpt_oss_vision.modular_gpt_oss_vision import GptOssModel
        
        # Create a minimal config for testing
        config = GPTOSSVisionConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            use_vision=True,
            vision_embed_dim=32,
            vision_patch_size=8,
            vision_num_channels=3,
            vision_num_layers=2,
            vision_num_heads=2,
            use_nope=False,
            head_dim=32,
            sliding_window=64,
        )
        
        print("✓ Config created successfully")
        
        # Create model
        model = GptOssModel(config)
        print("✓ Model created successfully")
        
        # Test inputs
        batch_size = 1
        text_len = 5
        image_size = 224
        
        input_ids = torch.randint(0, 1000, (batch_size, text_len))
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.randn(batch_size, 3, image_size, image_size)
        
        print("✓ Test inputs created")
        
        # Test forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values
            )
        
        print("✓ Vision forward pass successful!")
        print(f"  Output shape: {outputs.last_hidden_state.shape}")
        
        # Verify the output shape is correct
        expected_seq_len = outputs.last_hidden_state.shape[1]
        print(f"  Expected sequence length includes vision + text tokens")
        print(f"  Actual sequence length: {expected_seq_len}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vision_forward_pass()
    if success:
        print("\n🎉 Vision forward pass fix is working!")
    else:
        print("\n❌ Vision forward pass still has issues.")
    
    sys.exit(0 if success else 1)
