"""
Complete example workflow for SAM2 to Detectron2 pipeline.
This script demonstrates the full pipeline from data validation to inference.
"""

import os
import logging
from pathlib import Path

# Import our custom modules
from data_processor import SAM2DataProcessor
from detectron2_pipeline import SAM2ToDetectron2Pipeline
# Note: inference.py should be imported as a module if needed

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_complete_pipeline():
    """
    Example of complete pipeline from SAM2 data to trained Detectron2 model.
    Adjust the paths according to your data structure.
    """
    
    # =============================================================================
    # CONFIGURATION - MODIFY THESE PATHS FOR YOUR DATA
    # =============================================================================
    
    # Input data paths (from your SAM2 pipeline)
    SAM2_JSON_PATH = "../sample_data/18B104_0725_0011_0640_Movie2/segmentation_coco.json"  # Your uploaded file
    IMAGES_DIR = "../sample_data/18B104_0725_0011_0640_Movie2/"  # Directory containing 00000.jpg, 00001.jpg, etc.
        # Test video for inference
    TEST_VIDEO_PATH = "../sample_data/18B104_0725_0011_0640_Movie2.mp4"
    # Output directories
    BASE_OUTPUT_DIR = "./pipeline_output"
    SPLIT_DATA_DIR = f"{BASE_OUTPUT_DIR}/split_dataset"
    TRAINING_OUTPUT_DIR = f"{BASE_OUTPUT_DIR}/detectron2_training"
    OUTPUT_VIDEO_PATH = f"{BASE_OUTPUT_DIR}/inference_result.mp4"
    # Training parameters
    NUM_CLASSES = None  # Auto-detect from data
    LEARNING_RATE = 0.00025
    MAX_ITERATIONS = 5000
    BATCH_SIZE = 2
    CONFIDENCE_THRESHOLD = 0.7
    

    
    # =============================================================================
    # STEP 1: VALIDATE AND PROCESS SAM2 DATA
    # =============================================================================
    
    logger.info("=" * 60)
    logger.info("STEP 1: Validating SAM2 COCO data")
    logger.info("=" * 60)
    
    processor = SAM2DataProcessor()
    
    # Validate the data
    stats = processor.validate_coco_format(SAM2_JSON_PATH, IMAGES_DIR)
    processor.print_validation_report(stats)
    
    if not stats['valid']:
        logger.error("Data validation failed! Please fix the issues and try again.")
        return False
    
    # Create visualizations
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    processor.visualize_dataset_stats(f"{BASE_OUTPUT_DIR}/dataset_stats.png")
    
    # =============================================================================
    # STEP 2: SPLIT DATASET INTO TRAIN/VALIDATION
    # =============================================================================
    
    logger.info("=" * 60)
    logger.info("STEP 2: Splitting dataset")
    logger.info("=" * 60)
    
    train_json, val_json = processor.split_dataset(
        json_path=SAM2_JSON_PATH,
        images_dir=IMAGES_DIR,
        output_dir=SPLIT_DATA_DIR,
        train_ratio=0.8,
        val_ratio=0.2,
        seed=42
    )
    
    # Get the corresponding image directories
    train_images_dir = f"{SPLIT_DATA_DIR}/train/images"
    val_images_dir = f"{SPLIT_DATA_DIR}/val/images"
    
    # =============================================================================
    # STEP 3: TRAIN DETECTRON2 MODEL
    # =============================================================================
    
    logger.info("=" * 60)
    logger.info("STEP 3: Training Detectron2 model")
    logger.info("=" * 60)
    
    # Initialize training pipeline
    pipeline = SAM2ToDetectron2Pipeline(TRAINING_OUTPUT_DIR)
    
    # Setup configuration
    pipeline.setup_config(
        train_json=train_json,
        train_images_dir=train_images_dir,
        val_json=val_json,
        val_images_dir=val_images_dir,
        num_classes=NUM_CLASSES,  # Auto-detect
        learning_rate=LEARNING_RATE,
        max_iter=MAX_ITERATIONS,
        batch_size=BATCH_SIZE
    )
    
    # Train the model
    model_path = pipeline.train_model()
    logger.info(f"Training completed! Model saved to: {model_path}")
    
    # =============================================================================
    # STEP 4: SETUP PREDICTOR AND RUN INFERENCE
    # =============================================================================
    
    logger.info("=" * 60)
    logger.info("STEP 4: Running inference")
    logger.info("=" * 60)
    
    # Setup predictor
    pipeline.setup_predictor(model_path, confidence_threshold=CONFIDENCE_THRESHOLD)
    
    # Run inference on test video (if provided)
    if os.path.exists(TEST_VIDEO_PATH):
        detections = pipeline.predict_on_video(
            video_path=TEST_VIDEO_PATH,
            output_path=OUTPUT_VIDEO_PATH,
            visualize=True,
            save_detections=True
        )
        
        if detections:
            total_detections = sum(len(frame["detections"]) for frame in detections)
            logger.info(f"Inference completed! Total detections: {total_detections}")
    else:
        logger.warning(f"Test video not found: {TEST_VIDEO_PATH}")
        logger.info("Skipping inference step. You can run inference later using:")
        logger.info(f"python inference.py --model {model_path} --input your_video.mp4 --output result.mp4 --num-classes {pipeline.cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    
    # =============================================================================
    # STEP 5: GENERATE SUMMARY REPORT
    # =============================================================================
    
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    
    summary = f"""
Pipeline Execution Summary:
==========================

Data Statistics:
- Total images: {stats['num_images']}
- Total annotations: {stats['num_annotations']}
- Number of categories: {stats['num_categories']}
- Category distribution: {dict(stats['category_distribution'])}

Training Configuration:
- Model: Faster R-CNN with ResNet-50 FPN
- Number of classes: {pipeline.cfg.MODEL.ROI_HEADS.NUM_CLASSES if pipeline.cfg else 'N/A'}
- Learning rate: {LEARNING_RATE}
- Max iterations: {MAX_ITERATIONS}
- Batch size: {BATCH_SIZE}

Output Files:
- Trained model: {model_path if 'model_path' in locals() else 'N/A'}
- Dataset statistics: {BASE_OUTPUT_DIR}/dataset_stats.png
- Split dataset: {SPLIT_DATA_DIR}
- Training logs: {TRAINING_OUTPUT_DIR}/training

Next Steps:
-----------
1. Evaluate model performance on validation set
2. Adjust hyperparameters if needed
3. Run inference on new videos using the trained model
4. Consider fine-tuning with additional data

Command for inference:
python inference.py --model {model_path if 'model_path' in locals() else 'MODEL_PATH'} \\
                   --input VIDEO_PATH \\
                   --output OUTPUT_PATH \\
                   --num-classes {pipeline.cfg.MODEL.ROI_HEADS.NUM_CLASSES if pipeline.cfg else 'NUM_CLASSES'} \\
                   --confidence {CONFIDENCE_THRESHOLD}
"""
    
    print(summary)
    
    # Save summary to file
    with open(f"{BASE_OUTPUT_DIR}/pipeline_summary.txt", 'w') as f:
        f.write(summary)
    
    logger.info("Pipeline completed successfully! Check the output directory for all results.")
    return True

def quick_inference_example():
    """
    Example of using a pre-trained model for quick inference.
    """
    logger.info("Quick inference example")
    
    # Paths to your trained model and test data
    MODEL_PATH = "path/to/your/trained/model_final.pth"
    INPUT_VIDEO = "path/to/test_video.mp4"
    OUTPUT_VIDEO = "output_with_detections.mp4"
    NUM_CLASSES = 5  # Adjust based on your model
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found: {MODEL_PATH}")
        return
    
    # Initialize pipeline for inference only
    pipeline = SAM2ToDetectron2Pipeline()
    
    # Setup basic config (minimal setup for inference)
    cfg = pipeline.cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    
    # Setup predictor
    pipeline.setup_predictor(MODEL_PATH, confidence_threshold=0.7)
    
    # Run inference
    detections = pipeline.predict_on_video(
        video_path=INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        visualize=True,
        save_detections=True
    )
    
    logger.info(f"Inference completed! Output saved to: {OUTPUT_VIDEO}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Run quick inference example
        quick_inference_example()
    else:
        # Run complete pipeline
        success = run_complete_pipeline()
        
        if success:
            print("\n🎉 Pipeline completed successfully!")
            print("Check the pipeline_output directory for all results.")
        else:
            print("\n❌ Pipeline failed. Check the logs for details.")
            sys.exit(1)