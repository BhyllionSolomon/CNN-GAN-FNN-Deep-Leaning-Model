"""
=============================================================================
  CNN OCCLUSION CLASSIFIER  —  PhD Thesis, Mr. Olagunju Korede Solomon
  Matric: 216882, University of Ibadan
  Task: Binary classification — Ripe (fully visible) vs Occluded tomato
=============================================================================
  FIXES APPLIED TO ORIGINAL CODE:
  1. Class weights added — handles imbalance (6,803 Ripe vs 10,670 Occluded)
  2. Precision/Recall metrics fixed — added class_id parameter
  3. GradCAM added — required for IJACSA explainability paper
  4. AUC-ROC metric added — standard for binary classification papers
  5. Training curves combined into one figure — cleaner for paper
  6. F1-Score added to evaluation — matches thesis metrics
  7. All paths confirmed to match your folder structure
=============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight

# Reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS  —  confirmed to match your folder structure
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.abspath(r"C:\..PhD Thesis\DataSet\Processed_Tomatoes")
SAVE_DIR  = os.path.abspath(r"C:\..PhD Thesis\DataSet\Trained_Models")

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")
TEST_DIR  = os.path.join(BASE_DIR, "test")

os.makedirs(SAVE_DIR, exist_ok=True)

IMG_HEIGHT  = 224
IMG_WIDTH   = 224
BATCH_SIZE  = 32
EPOCHS      = 50
CLASS_NAMES = ["Occluded", "Ripe"]   # alphabetical — Keras auto-sorts


# ═════════════════════════════════════════════════════════════════════════════
#  1. DATA GENERATORS
# ═════════════════════════════════════════════════════════════════════════════
def create_data_generators():
    # Training: augmentation only (no synthetic — pipeline already augmented)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Val / Test: only rescale
    eval_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    print("=" * 60)
    print("PATH VERIFICATION")
    print("=" * 60)
    for name, path in [("Train", TRAIN_DIR), ("Val", VAL_DIR), ("Test", TEST_DIR)]:
        exists = os.path.exists(path)
        print(f"{name}: {path}  |  Exists: {exists}")
        if exists:
            subdirs = os.listdir(path)
            for sd in subdirs:
                count = len(os.listdir(os.path.join(path, sd)))
                print(f"       {sd}/  →  {count} images")
    print("=" * 60)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )
    val_gen = eval_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    test_gen = eval_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    print(f"\nClasses detected : {train_gen.class_indices}")
    print(f"Training samples : {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples     : {test_gen.samples}")

    return train_gen, val_gen, test_gen


# ═════════════════════════════════════════════════════════════════════════════
#  2. CLASS WEIGHTS  (fixes imbalance: 6,803 Ripe vs 10,670 Occluded)
# ═════════════════════════════════════════════════════════════════════════════

def get_class_weights(train_gen):
    labels = train_gen.classes
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(weights))
    print(f"\nClass weights (balancing imbalance):")
    for class_name, class_index in train_gen.class_indices.items():  # FIXED: class_name is string, class_index is int
        print(f"  Class {class_name} (index {class_index}): weight = {class_weight_dict[class_index]:.4f}")
    return class_weight_dict

# ═════════════════════════════════════════════════════════════════════════════
#  3. CNN MODEL  (matches thesis architecture exactly)
# ═════════════════════════════════════════════════════════════════════════════
def build_model():
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='input')

    # Block 1 — low-level features (edges, colour gradients)
    x = layers.Conv2D(32, (3,3), padding='same',
                      kernel_regularizer=regularizers.l2(0.001),
                      name='conv1')(inputs)
    x = layers.ReLU(name='relu1')(x)
    x = layers.MaxPooling2D((2,2), name='pool1')(x)

    # Block 2 — mid-level features (textures, shapes)
    x = layers.Conv2D(64, (3,3), padding='same',
                      kernel_regularizer=regularizers.l2(0.001),
                      name='conv2')(x)
    x = layers.ReLU(name='relu2')(x)
    x = layers.MaxPooling2D((2,2), name='pool2')(x)

    # Block 3 — high-level features (occlusion patterns)
    x = layers.Conv2D(128, (3,3), padding='same',
                      kernel_regularizer=regularizers.l2(0.001),
                      name='conv3')(x)
    x = layers.ReLU(name='relu3')(x)
    x = layers.MaxPooling2D((2,2), name='pool3')(x)

    # Global pooling + classifier head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001),
                     name='dense_128')(x)
    x = layers.Dropout(0.4, name='dropout')(x)

    # Output: 2 classes with softmax
    outputs = layers.Dense(2, activation='softmax', name='output')(x)

    model = models.Model(inputs, outputs, name='Tomato_CNN_Classifier')
    return model


# ═════════════════════════════════════════════════════════════════════════════
#  4. TRAINING
# ═════════════════════════════════════════════════════════════════════════════
def train_model(model, train_gen, val_gen, class_weights):
    # FIX: Precision/Recall need class_id for binary task with softmax output
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(class_id=1, name='precision'),
            tf.keras.metrics.Recall(class_id=1, name='recall'),
            tf.keras.metrics.AUC(name='auc')        # added for paper
        ]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(SAVE_DIR, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            os.path.join(SAVE_DIR, 'training_log.csv')
        )
    ]

    print(f"\nTraining with class weights: {class_weights}")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,   # FIX: handles imbalance
        verbose=1
    )
    return history


# ═════════════════════════════════════════════════════════════════════════════
#  5. TRAINING CURVES  (single combined figure — suitable for paper)
# ═════════════════════════════════════════════════════════════════════════════
def plot_training_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CNN Training History — Tomato Occlusion Classifier',
                 fontsize=16, fontweight='bold')

    metrics = [
        ('accuracy',  'val_accuracy',  'Accuracy',  axes[0,0]),
        ('loss',      'val_loss',      'Loss',      axes[0,1]),
        ('precision', 'val_precision', 'Precision', axes[1,0]),
        ('recall',    'val_recall',    'Recall',    axes[1,1]),
    ]

    for train_m, val_m, title, ax in metrics:
        if train_m in history.history:
            ax.plot(history.history[train_m], label='Train', linewidth=2)
        if val_m in history.history:
            ax.plot(history.history[val_m],   label='Val',   linewidth=2)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, 'training_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved: {path}")


# ═════════════════════════════════════════════════════════════════════════════
#  6. EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
def evaluate_model(model, test_gen):
    print("\n" + "=" * 60)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 60)

    y_prob  = model.predict(test_gen, verbose=1)
    y_pred  = np.argmax(y_prob, axis=1)
    y_true  = test_gen.classes
    classes = list(test_gen.class_indices.keys())

    # Classification report
    report = classification_report(y_true, y_pred,
                                   target_names=classes, digits=4)
    print("\nClassification Report:")
    print(report)

    # Save report
    rpt_path = os.path.join(SAVE_DIR, 'classification_report.txt')
    with open(rpt_path, 'w') as f:
        f.write("TOMATO CNN CLASSIFIER — OCCLUSION DETECTION\n")
        f.write("PhD Thesis: Olagunju Korede Solomon (216882)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test samples : {test_gen.samples}\n")
        f.write(f"Classes      : {test_gen.class_indices}\n\n")
        f.write(report)
    print(f"Report saved: {rpt_path}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                annot_kws={"size": 14})
    plt.title('Confusion Matrix', fontsize=15, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    cm_path = os.path.join(SAVE_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved: {cm_path}")

    # ROC curve (important for paper)
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc     = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve — Occlusion Detection', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(SAVE_DIR, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"ROC curve saved: {roc_path}")

    return y_prob, y_true, y_pred


# ═════════════════════════════════════════════════════════════════════════════
#  7. GRADCAM  (required for IJACSA explainability paper)
# ═════════════════════════════════════════════════════════════════════════════
def make_gradcam_heatmap(model, img_array, last_conv_layer='conv3'):
    """Generate GradCAM heatmap for a single image."""
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads      = tape.gradient(class_channel, conv_outputs)
    pooled     = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out   = conv_outputs[0]
    heatmap    = conv_out @ pooled[..., tf.newaxis]
    heatmap    = tf.squeeze(heatmap)
    heatmap    = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index.numpy()


def save_gradcam(model, test_gen, num_samples=6):
    """Save GradCAM overlays for a sample of test images."""
    print("\nGenerating GradCAM visualisations...")
    images, labels = next(iter(test_gen))
    classes = list(test_gen.class_indices.keys())

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 7))
    fig.suptitle('GradCAM — CNN Attention on Tomato Features',
                 fontsize=14, fontweight='bold')

    for i in range(num_samples):
        img        = images[i]
        img_input  = np.expand_dims(img, axis=0)
        true_idx   = np.argmax(labels[i])

        heatmap, pred_idx = make_gradcam_heatmap(model, img_input)

        # Resize heatmap to image size
        heatmap_resized = np.uint8(255 * heatmap)
        heatmap_resized = np.array(
            tf.image.resize(heatmap_resized[..., np.newaxis],
                            (IMG_HEIGHT, IMG_WIDTH))
        ).squeeze()

        # Apply colormap
        colormap   = cm.get_cmap('jet')
        heatmap_rgb = colormap(heatmap_resized / 255.0)[:, :, :3]
        overlay    = 0.5 * img + 0.5 * heatmap_rgb
        overlay    = np.clip(overlay, 0, 1)

        # Original
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"True: {classes[true_idx]}", fontsize=9)

        # GradCAM
        axes[1, i].imshow(overlay)
        axes[1, i].axis('off')
        color = 'green' if pred_idx == true_idx else 'red'
        axes[1, i].set_title(f"Pred: {classes[pred_idx]}",
                              fontsize=9, color=color)

    plt.tight_layout()
    gcam_path = os.path.join(SAVE_DIR, 'gradcam_visualisation.png')
    plt.savefig(gcam_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"GradCAM saved: {gcam_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  8. PREDICTION SAMPLES
# ═════════════════════════════════════════════════════════════════════════════
def visualise_predictions(model, test_gen, num_samples=6):
    images, labels = next(iter(test_gen))
    preds      = model.predict(images)
    pred_idx   = np.argmax(preds, axis=1)
    true_idx   = np.argmax(labels, axis=1)
    classes    = list(test_gen.class_indices.keys())

    plt.figure(figsize=(15, 6))
    for i in range(min(num_samples, len(images))):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i])
        plt.axis('off')
        color = 'green' if pred_idx[i] == true_idx[i] else 'red'
        plt.title(
            f"True:  {classes[true_idx[i]]}\n"
            f"Pred:  {classes[pred_idx[i]]} ({preds[i][pred_idx[i]]:.2f})",
            color=color, fontsize=10
        )
    plt.suptitle('Prediction Examples — Ripe vs Occluded',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    pred_path = os.path.join(SAVE_DIR, 'prediction_examples.png')
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Prediction examples saved: {pred_path}")


# ═════════════════════════════════════════════════════════════════════════════
#  9. MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  TOMATO CNN OCCLUSION CLASSIFIER")
    print("  PhD Thesis — Olagunju Korede Solomon (216882)")
    print("=" * 60)

    try:
        # Step 1: Load data
        print("\n[1/7] Loading data generators...")
        train_gen, val_gen, test_gen = create_data_generators()

        # Step 2: Compute class weights
        print("\n[2/7] Computing class weights...")
        class_weights = get_class_weights(train_gen)

        # Step 3: Build model
        print("\n[3/7] Building CNN model...")
        model = build_model()

        # Step 4: Train
        print("\n[4/7] Training...")
        history = train_model(model, train_gen, val_gen, class_weights)

        # Step 5: Plot training curves
        print("\n[5/7] Plotting training history...")
        plot_training_history(history)

        # Step 6: Evaluate on test set
        print("\n[6/7] Evaluating on test set...")
        evaluate_model(model, test_gen)

        # Step 7: GradCAM + predictions
        print("\n[7/7] Generating GradCAM and prediction samples...")
        save_gradcam(model, test_gen)
        visualise_predictions(model, test_gen)

        # Save final model
        final_path = os.path.join(SAVE_DIR, 'tomato_cnn_final.h5')
        model.save(final_path)
        print(f"\nFinal model saved: {final_path}")

        print("\n" + "=" * 60)
        print("  TRAINING COMPLETE")
        print(f"  All outputs saved to: {SAVE_DIR}")
        print("=" * 60)

    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()