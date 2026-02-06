"""
Automated CNN Training Script (no user input required)
"""
import cv2
import numpy as np
import sys
import os

sys.path.insert(0, 'api')
from cnn_antispoofing import CNNAntiSpoofing

def load_existing_images():
    """Load images from training_data folder"""
    real_images = []
    fake_images = []
    
    real_dir = 'training_data/real'
    if os.path.exists(real_dir):
        for filename in os.listdir(real_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(real_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    real_images.append(img_rgb)
    
    fake_dir = 'training_data/fake'
    if os.path.exists(fake_dir):
        for filename in os.listdir(fake_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(fake_dir, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    fake_images.append(img_rgb)
    
    return real_images, fake_images

def main():
    print("\n" + "="*70)
    print("CNN ANTI-SPOOFING TRAINING (Automated)")
    print("="*70)
    
    # Load training data
    existing_real, existing_fake = load_existing_images()
    
    if len(existing_real) == 0 or len(existing_fake) == 0:
        print("\n[ERROR] No training data found!")
        print("Please collect training data using the web interface at: http://localhost:8000/train.html")
        return
    
    print(f"\n[OK] Found training data:")
    print(f"   Real images: {len(existing_real)}")
    print(f"   Fake images: {len(existing_fake)}")
    
    if len(existing_real) < 20 or len(existing_fake) < 20:
        print(f"\n[WARNING] Limited training data!")
        print(f"   Recommended: At least 30 of each type")
        print(f"   Current: {len(existing_real)} real, {len(existing_fake)} fake")
        print("\n   Proceeding anyway...")
    
    # Training parameters
    epochs = 20
    batch_size = 16
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Model: MobileNetV2 + Custom Head")
    print(f"  Input size: 224x224x3")
    
    # Train model
    print("\n" + "="*70)
    print("TRAINING CNN MODEL")
    print("="*70)
    
    cnn_spoof = CNNAntiSpoofing()
    
    try:
        metrics = cnn_spoof.train(
            existing_real, 
            existing_fake, 
            save_images=False,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print(f"[OK] Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"[OK] Precision: {metrics['precision']*100:.2f}%")
        print(f"[OK] Recall: {metrics['recall']*100:.2f}%")
        print(f"[OK] F1 Score: {metrics['f1_score']*100:.2f}%")
        print(f"[OK] Model saved to models/cnn_antispoofing_model.h5")
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("1. Restart your server: python api/main.py")
        print("2. Test with real and fake images")
        print("3. The CNN model will be used automatically")
        print("\nYour system now has state-of-the-art anti-spoofing!")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
