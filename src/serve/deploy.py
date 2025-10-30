"""
Ray Serve deployment script
"""

import ray
from ray import serve
import argparse
import logging
from pathlib import Path
import yaml

from src.serve.app import MultitaskSignalModel
from src.utils.logger import setup_logging, load_config


def main():
    parser = argparse.ArgumentParser(description='Deploy Ray Serve application')
    parser.add_argument('--config', type=str, default='conf/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind to')
    parser.add_argument('--num-replicas', type=int, default=2,
                       help='Number of model replicas')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logger = setup_logging(
        log_level=config['logging']['level'],
        log_dir=config['logging']['log_dir']
    )
    
    logger.info("Starting Ray Serve deployment...")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Replicas: {args.num_replicas}")
    
    # Initialize Ray
    ray.init()
    
    # Update config with checkpoint if provided
    if args.checkpoint:
        config['model']['checkpoint_path'] = args.checkpoint
    
    # Create deployment
    deployment = MultitaskSignalModel.options(
        num_replicas=args.num_replicas,
        ray_actor_options={"num_cpus": 1, "num_gpus": 0.5}
    ).bind(config['model']['checkpoint_path'] if 'checkpoint_path' in config.get('model', {}) else args.config)
    
    # Deploy
    serve.run(
        deployment,
        host=args.host,
        port=args.port,
        name="multitask_signal_model"
    )
    
    logger.info("Ray Serve deployment completed!")
    logger.info(f"API available at: http://{args.host}:{args.port}")
    
    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()


