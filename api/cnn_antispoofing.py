"""
CNN-Based Anti-Spoofing System
Uses deep learning for robust face liveness detection
"""
import cv2
import numpy as np
import os
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

logger = logging.getLogger(__name__)

class CNNAntiSpoofing:
    def __init__(self, model_path='models/cnn_antispoofing_model.h5'):
        self.model_path = model_path
        self.model = None
        self.input_shape = (224, 224, 3)
        self.training_data_dir = 'training_data'
        self.real_dir = os.path.join(self.training_data_dir, 'real')
        self.fake_dir = os.path.join(self.training_data_dir, 'fake')
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self._ensure_directories()
        self.load_model()
    
    def _ensure_directories(self):
        """Create training data directories if they don't exist"""
        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.fake_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)
    
    def build_model(self):
        """Build CNN model using transfer learning with MobileNetV2"""
        logger.info("Building CNN model with MobileNetV2 backbone...")
        
        # Use MobileNetV2 as backbone (pre-trained on ImageNet)
        base_model = MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Build model
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # Data augmentation layers (applied during training)
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            
            # Pre-trained backbone
            base_model,
            
            # Custom classification head
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            # Output layer (binary classification: real vs fake)
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
        )
        
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)
        
        return model
    
    def extract_face(self, img_array):
        """Extract and preprocess face region"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 3)
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            padding = int(w * 0.2)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img_array.shape[1] - x, w + 2*padding)
            h = min(img_array.shape[0] - y, h + 2*padding)
            return img_array[y:y+h, x:x+w]
        return img_array
    
    def preprocess_image(self, img_array):
        """Preprocess image for CNN input"""
        # Extract face
        face_img = self.extract_face(img_array)
        
        # Resize to model input size
        img_resized = cv2.resize(face_img, (self.input_shape[0], self.input_shape[1]))
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype('float32') / 255.0
        
        return img_normalized
    
    def save_training_images(self, real_images, fake_images):
        """Save training images to disk"""
        saved_real = 0
        saved_fake = 0
        
        for i, img in enumerate(real_images):
            try:
                face_img = self.extract_face(img)
                filename = f"real_{int(datetime.now().timestamp())}_{i}.jpg"
                filepath = os.path.join(self.real_dir, filename)
                cv2.imwrite(filepath, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                saved_real += 1
            except Exception as e:
                logger.error(f"Failed to save real image {i}: {e}")
        
        for i, img in enumerate(fake_images):
            try:
                face_img = self.extract_face(img)
                filename = f"fake_{int(datetime.now().timestamp())}_{i}.jpg"
                filepath = os.path.join(self.fake_dir, filename)
                cv2.imwrite(filepath, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                saved_fake += 1
            except Exception as e:
                logger.error(f"Failed to save fake image {i}: {e}")
        
        return saved_real, saved_fake
    
    def load_training_images(self):
        """Load all saved training images from disk"""
        real_images = []
        fake_images = []
        
        for filename in os.listdir(self.real_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.real_dir, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    real_images.append(img_rgb)
        
        for filename in os.listdir(self.fake_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                filepath = os.path.join(self.fake_dir, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    fake_images.append(img_rgb)
        
        return real_images, fake_images
    
    def get_training_stats(self):
        """Get statistics about saved training data"""
        real_count = len([f for f in os.listdir(self.real_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        fake_count = len([f for f in os.listdir(self.fake_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        return real_count, fake_count
    
    def train(self, real_images, fake_images, save_images=True, epochs=20, batch_size=16):
        """Train the CNN model"""
        # Save new images if requested
        if save_images:
            logger.info("Saving training images...")
            saved_real, saved_fake = self.save_training_images(real_images, fake_images)
            logger.info(f"Saved {saved_real} real and {saved_fake} fake face images")
        
        # Load all existing training data
        logger.info("\nLoading all training data from disk...")
        all_real, all_fake = self.load_training_images()
        logger.info(f"Total training data: {len(all_real)} real, {len(all_fake)} fake images")
        
        if len(all_real) < 20 or len(all_fake) < 20:
            raise ValueError("Need at least 20 real and 20 fake images to train CNN")
        
        # Prepare data
        logger.info("\nPreprocessing images...")
        X = []
        y = []
        
        for i, img in enumerate(all_real):
            if i % 10 == 0:
                logger.info(f"Processing real image {i+1}/{len(all_real)}")
            preprocessed = self.preprocess_image(img)
            X.append(preprocessed)
            y.append(1)  # 1 = real
        
        for i, img in enumerate(all_fake):
            if i % 10 == 0:
                logger.info(f"Processing fake image {i+1}/{len(all_fake)}")
            preprocessed = self.preprocess_image(img)
            X.append(preprocessed)
            y.append(0)  # 0 = fake
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"\nTraining with {len(all_real)} real and {len(all_fake)} fake images")
        logger.info(f"Input shape: {X.shape}")
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} images")
        logger.info(f"Test set: {len(X_test)} images")
        
        # Build model
        self.model = self.build_model()
        
        # Custom callback to save progress
        class ProgressCallback(keras.callbacks.Callback):
            def __init__(self, progress_file='training_progress.json'):
                super().__init__()
                self.progress_file = progress_file
            
            def on_epoch_end(self, epoch, logs=None):
                # Write progress with actual metrics after epoch completes
                progress = {
                    'epoch': epoch + 1,
                    'total_epochs': self.params['epochs'],
                    'loss': float(logs.get('loss', 0)),
                    'accuracy': float(logs.get('accuracy', 0)),
                    'val_loss': float(logs.get('val_loss', 0)),
                    'val_accuracy': float(logs.get('val_accuracy', 0)),
                    'precision': float(logs.get('precision', 0)),
                    'recall': float(logs.get('recall', 0)),
                    'status': 'training'
                }
                # Write to file for frontend to poll
                import json
                with open(self.progress_file, 'w') as f:
                    json.dump(progress, f)
                
                # Log to console
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch + 1}/{self.params['epochs']} Complete")
                logger.info(f"{'='*60}")
                logger.info(f"Loss: {logs.get('loss', 0):.4f} - Acc: {logs.get('accuracy', 0)*100:.1f}%")
                logger.info(f"Val Loss: {logs.get('val_loss', 0):.4f} - Val Acc: {logs.get('val_accuracy', 0)*100:.1f}%")
                logger.info(f"Precision: {logs.get('precision', 0):.4f} - Recall: {logs.get('recall', 0):.4f}")
                logger.info(f"{'='*60}\n")
        
        # Callbacks
        callbacks = [
            ProgressCallback(),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        # Train model
        logger.info("\nTraining CNN model...")
        logger.info(f"{'='*60}")
        logger.info(f"Starting training with {epochs} epochs, batch size {batch_size}")
        logger.info(f"{'='*60}\n")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2  # verbose=2 shows one line per epoch with metrics
        )
        
        # Evaluate
        logger.info("\nEvaluating model...")
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Predictions for confusion matrix
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        from sklearn.metrics import confusion_matrix, f1_score
        cm = confusion_matrix(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Complete!")
        logger.info(f"{'='*60}")
        logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
        logger.info(f"Test Precision: {test_precision*100:.2f}%")
        logger.info(f"Test Recall: {test_recall*100:.2f}%")
        logger.info(f"F1 Score: {f1*100:.2f}%")
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"                Predicted")
        logger.info(f"              Fake  Real")
        logger.info(f"Actual Fake   {cm[0][0]:4d}  {cm[0][1]:4d}")
        logger.info(f"       Real   {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        # Save model
        self.save_model()
        
        return {
            'accuracy': float(test_acc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'total_real': len(all_real),
            'total_fake': len(all_fake),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def predict(self, img_array, confidence_threshold=0.90):
        """Predict if image is real or fake using CNN
        
        Args:
            img_array: Input image
            confidence_threshold: Minimum confidence to accept as real (default 0.90 = 90%)
        """
        if self.model is None:
            return False, "CNN model not trained. Please train first.", 0
        
        # Preprocess image
        preprocessed = self.preprocess_image(img_array)
        
        # Add batch dimension
        img_batch = np.expand_dims(preprocessed, axis=0)
        
        # Predict
        prediction = self.model.predict(img_batch, verbose=0)[0][0]
        
        # prediction is probability of being real (0.0 to 1.0)
        real_probability = float(prediction)
        
        logger.info(f"[CNN PREDICT] Real probability: {real_probability:.4f}")
        logger.info(f"[CNN PREDICT] Threshold: {confidence_threshold:.2f}")
        logger.info(f"[CNN PREDICT] Decision: {real_probability >= confidence_threshold}")
        
        is_real = bool(real_probability >= confidence_threshold)
        confidence = int(real_probability * 100)
        
        if is_real:
            message = f"Live person verified (CNN confidence: {confidence}%)"
        else:
            message = f"Spoof detected (CNN confidence: {100-confidence}%)"
        
        return is_real, message, confidence
    
    def save_model(self):
        """Save trained model to disk"""
        self.model.save(self.model_path)
        logger.info(f"\n[OK] CNN model saved to {self.model_path}")
    
    def load_model(self):
        """Load trained model from disk"""
        if os.path.exists(self.model_path):
            try:
                self.model = keras.models.load_model(self.model_path)
                logger.info(f"[OK] CNN model loaded from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"[ERROR] Error loading CNN model: {e}")
                return False
        return False
    
    def is_trained(self):
        """Check if model is trained"""
        return self.model is not None
