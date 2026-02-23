"""Run maturation phase: 100,000 steps to train a mature Cogito agent."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from cogito.config import Config
from cogito.core.simulation import Simulation
from cogito.monitoring.data_collector import DataCollector
from cogito.monitoring.state_analyzer import StateAnalyzer
from cogito.monitoring.svc_detector import SVCDetector
from cogito.monitoring.dashboard import Dashboard


def run_maturation(
    num_steps: int = 100000,
    checkpoint_interval: int = 1000,
    snapshot_interval: int = 5000,
    analysis_interval: int = 500,
    print_interval: int = 10000,
    headless: bool = True,
) -> dict:
    """Run maturation phase simulation.
    
    Args:
        num_steps: Total steps to run (default 100,000)
        checkpoint_interval: Steps between weight saves
        snapshot_interval: Steps between dashboard snapshots
        analysis_interval: Steps between state analysis
        print_interval: Steps between detailed stats print
        headless: Whether to run without display
        
    Returns:
        Dictionary with final statistics
    """
    print("=" * 60)
    print("Cogito Maturation Phase")
    print("=" * 60)
    print(f"Target steps: {num_steps:,}")
    print(f"Checkpoint interval: {checkpoint_interval:,}")
    print(f"Snapshot interval: {snapshot_interval:,}")
    print(f"Analysis interval: {analysis_interval:,}")
    print()
    
    # Create directories
    Config.create_dirs()
    
    # Ensure snapshot directory exists
    snapshot_dir = Path("data/snapshots")
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize simulation
    print("Initializing simulation...")
    sim = Simulation(headless=headless)
    
    # Initialize monitoring
    print("Initializing monitoring...")
    collector = DataCollector(Config)
    analyzer = StateAnalyzer(Config)
    svc_detector = SVCDetector(Config)
    dashboard = Dashboard(Config, headless=headless)
    
    # Tracking variables
    start_time = time.time()
    stats_history = []
    svc_history = []
    
    # Register collector as callback
    def step_callback(step: int, agent, world, info: dict):
        collector.collect(step, agent, world, info)
    
    def analysis_callback(step: int, agent, world, info: dict):
        nonlocal svc_history
        
        # Run analysis at intervals
        if step > 0 and step % analysis_interval == 0:
            # Get recent internal states
            states = collector.get_internal_states(
                max(0, step - analysis_interval), step
            )
            if states is not None and len(states) >= 50:
                # Get behavior data for correlation
                behavior = collector.get_behavior_stats(analysis_interval)
                
                # Run analysis
                result = analyzer.analyze(states, behavior)
                
                # Run SVC detection
                report = svc_detector.detect(result, behavior, step)
                svc_detector.update_history(report)
                svc_history.append({
                    'step': step,
                    'is_detected': report.is_detected,
                    'confidence': report.confidence,
                    'num_candidates': len(report.candidate_clusters),
                })
                
                # Update dashboard
                if not headless:
                    dashboard.update(step, world, agent, result, report)
    
    # Combined callback
    def callback(step: int, agent, world, info: dict):
        step_callback(step, agent, world, info)
        analysis_callback(step, agent, world, info)
    
    # Run simulation in chunks for better memory management
    chunk_size = 10000
    if num_steps < chunk_size:
        chunk_size = num_steps
        chunks = 1
    else:
        chunks = num_steps // chunk_size
    
    print(f"Running {chunks} chunks of {chunk_size:,} steps...")
    print()
    
    for chunk in range(chunks):
        chunk_start = chunk * chunk_size
        
        # Run chunk
        sim.run(chunk_size, callbacks=[callback])
        
        # Save checkpoint
        current_step = (chunk + 1) * chunk_size
        if current_step % checkpoint_interval == 0:
            checkpoint_path = Path(f"data/checkpoints/step_{current_step:06d}.pt")
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            sim.agent.save(str(checkpoint_path))
        
        # Save dashboard snapshot
        if current_step % snapshot_interval == 0:
            snapshot_path = snapshot_dir / f"snapshot_{current_step:06d}.png"
            dashboard.save_snapshot(str(snapshot_path))
        
        # Print progress
        if current_step % print_interval == 0 or chunk == 0:
            elapsed = time.time() - start_time
            stats = collector.get_behavior_stats(1000)
            
            # Calculate averages
            avg_energy = stats.get('avg_energy', 0)
            avg_lifespan = stats.get('avg_lifespan', 0)
            deaths = stats.get('deaths', 0)
            food_eaten = stats.get('food_eaten', 0)
            
            # Get learning stats
            learning_curve = collector.get_learning_curve(1000)
            avg_pred_loss = learning_curve.get('avg_prediction_loss', 0)
            
            # Estimate remaining time
            steps_done = current_step
            steps_left = num_steps - steps_done
            time_per_step = elapsed / max(1, steps_done)
            remaining_time = steps_left * time_per_step / 60  # minutes
            
            print(f"Step {current_step:6d} | "
                  f"Energy: {avg_energy:5.1f} | "
                  f"Lifespan: {avg_lifespan:5.1f} | "
                  f"Deaths: {deaths:4d} | "
                  f"Food: {food_eaten:4d} | "
                  f"PredLoss: {avg_pred_loss:.4f} | "
                  f"ETA: {remaining_time:.1f}m")
            
            stats_history.append({
                'step': current_step,
                'avg_energy': avg_energy,
                'avg_lifespan': avg_lifespan,
                'deaths': deaths,
                'food_eaten': food_eaten,
                'prediction_loss': avg_pred_loss,
                'elapsed_time': elapsed,
            })
    
    # Final summary
    print()
    print("=" * 60)
    print("Maturation Complete!")
    print("=" * 60)
    
    total_time = time.time() - start_time
    final_stats = collector.get_behavior_stats(10000)
    final_learning = collector.get_learning_curve(10000)
    
    print(f"Total time: {total_time / 60:.1f} minutes")
    if total_time > 0:
        print(f"Steps per second: {num_steps / total_time:.1f}")
    print()
    print("Final Statistics (last 10,000 steps):")
    print(f"  Average energy: {final_stats.get('avg_energy', 0):.1f}")
    print(f"  Average lifespan: {final_stats.get('avg_lifespan', 0):.1f}")
    print(f"  Total deaths: {final_stats.get('total_deaths', 0)}")
    print(f"  Food eaten: {final_stats.get('food_eaten', 0)}")
    print(f"  Prediction loss: {final_learning.get('avg_prediction_loss', 0):.4f}")
    print()
    
    # Learning improvement check
    if len(stats_history) >= 2:
        initial_lifespan = stats_history[0].get('avg_lifespan', 1)
        final_lifespan = stats_history[-1].get('avg_lifespan', 1)
        improvement = final_lifespan / max(1, initial_lifespan)
        print(f"Lifespan improvement: {improvement:.1f}x")
        
        initial_loss = stats_history[0].get('prediction_loss', 1)
        final_loss = stats_history[-1].get('prediction_loss', 1)
        loss_reduction = final_loss / max(0.001, initial_loss)
        print(f"Prediction loss ratio: {loss_reduction:.2f}")
    
    print()
    
    # SVC summary
    if svc_history:
        detected_count = sum(1 for h in svc_history if h['is_detected'])
        avg_confidence = sum(h['confidence'] for h in svc_history) / len(svc_history)
        print("SVC Detection Summary:")
        print(f"  Analyses performed: {len(svc_history)}")
        print(f"  Detections: {detected_count}")
        print(f"  Average confidence: {avg_confidence:.2f}")
    
    # Close resources
    collector.close()
    dashboard.close()
    
    # Save final checkpoint
    final_checkpoint = Path("data/checkpoints/final_mature.pt")
    sim.agent.save(str(final_checkpoint))
    print(f"\nFinal checkpoint saved to: {final_checkpoint}")
    
    return {
        'total_time': total_time,
        'stats_history': stats_history,
        'svc_history': svc_history,
        'final_stats': final_stats,
        'final_learning': final_learning,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Cogito maturation phase")
    parser.add_argument(
        "--steps", type=int, default=100000,
        help="Number of steps to run (default: 100000)"
    )
    parser.add_argument(
        "--checkpoint-interval", type=int, default=1000,
        help="Steps between checkpoints"
    )
    parser.add_argument(
        "--snapshot-interval", type=int, default=5000,
        help="Steps between dashboard snapshots"
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show visualization (default: headless)"
    )
    
    args = parser.parse_args()
    
    run_maturation(
        num_steps=args.steps,
        checkpoint_interval=args.checkpoint_interval,
        snapshot_interval=args.snapshot_interval,
        headless=not args.display,
    )
