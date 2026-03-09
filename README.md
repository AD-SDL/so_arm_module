# SO-Arm + LeRobot Complete Guide

## Table of Contents
1. [Initial Setup](#initial-setup)
2. [Motor Calibration](#motor-calibration)
3. [Basic Teleoperation](#basic-teleoperation)
4. [Teleoperation with Camera](#teleoperation-with-camera)
5. [Data Collection](#data-collection)
6. [Training](#training)
7. [Deployment](#deployment)

---

## Initial Setup

### Hardware Connections
```bash
# Check USB devices
ls -l /dev/ttyACM*

# Fix permissions (required after each reboot)
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

# Permanent fix (add user to dialout group)
sudo usermod -a -G dialout $USER
# Then log out and log back in
```

### Identify Controllers
```bash
# Leader arm typically on /dev/ttyACM0
# Follower arm typically on /dev/ttyACM1
# Verify by unplugging one at a time and checking ls -l /dev/ttyACM*
```

### Camera Setup
```bash
# Find cameras
lerobot-find-cameras

# For RealSense camera, note the serial number (e.g., 327122076093)
```

---

## Motor Calibration

### Why Calibrate?
Calibration maps motor positions to actual joint angles. Required once per robot, but must be redone if:
- Robot is moved to a new machine
- Calibration files are deleted
- Motors are replaced

### Leader Arm Calibration
```bash
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader_arm1
```

**Process:**
1. Manually position arm to middle of range of motion
2. Press ENTER
3. Move each joint through full range
4. Press ENTER when done
5. Calibration saved to `~/.cache/huggingface/lerobot/calibration/`

### Follower Arm Calibration
```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_arm1
```

**Same process as leader arm.**

### Troubleshooting Calibration
- **"Magnitude exceeds 2047" error:** Arm joints positioned too far from center. Manually move all joints to mid-range before starting.
- **Voltage errors/red flashing:** Check power supply voltage (leader: 5V, follower: 12V)
- **Communication errors:** Power cycle the arm (unplug USB + power, wait 10 seconds, reconnect)

---

## Basic Teleoperation

### Command
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_arm1 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader_arm1
```

### Best Practices
- Position both arms similarly before starting to avoid sudden movements
- Start with slow, gentle movements to test
- If collision occurs, power cycle the follower arm
- Keep clear space around both arms

---

## Teleoperation with Camera

### RealSense Camera
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_arm1 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader_arm1 \
    --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 327122076093, width: 640, height: 480, fps: 30}}" \
    --display_data=true
```

### Regular Webcam
```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_arm1 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader_arm1 \
    --robot.cameras="{ front: {type: opencv, index_or_path: /dev/video4, width: 1280, height: 720, fps: 30}}" \
    --display_data=true
```

### Camera Setup Notes
- Replace `327122076093` with your RealSense serial number
- Replace `/dev/video4` with your webcam path
- Recommended resolution: 640x480 @ 30fps (good balance of quality and training speed)

---

## Data Collection

### Basic Recording Command
```bash
export HF_HUB_OFFLINE=1

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=follower_arm1 \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=leader_arm1 \
    --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 327122076093, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=local/my_dataset \
    --dataset.single_task="Pick and place cube" \
    --dataset.fps=30 \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=10 \
    --dataset.push_to_hub=false \
    --display_data=true
```

### Resuming Data Collection
```bash
# Add --resume=true to continue adding episodes
lerobot-record \
    [... same parameters as above ...] \
    --resume=true
```

### Recording Workflow
1. **"Recording episode X"**: Leader controls follower - perform your task
2. **"Reset the environment"**: Follower stops - manually reset workspace
3. **Automatic start**: Next episode begins after reset time
4. **Keyboard controls**:
   - Right Arrow (→): Skip to next episode early
   - Left Arrow (←): Cancel current episode and re-record
   - Escape (ESC): Stop recording completely

### Dataset Location
```bash
# Data saved to:
~/.cache/huggingface/lerobot/local/my_dataset/

# Check dataset structure:
ls -la ~/.cache/huggingface/lerobot/local/my_dataset/
# Should see: data/, meta/, videos/
```

---

## Data Collection Best Practices

### Camera Positioning
**Critical:** Camera position must be IDENTICAL between data collection and deployment.

**Recommended Setup:**
- **Distance:** 18-24 inches from workspace
- **Height:** 8-12 inches above table surface
- **Angle:** 30-45 degrees downward
- **Field of view must include:**
  - All object starting positions
  - Target placement location (bowl/box)
  - Robot gripper during manipulation
  - Entire workspace area

**Tips:**
- Mark camera mount position with tape
- Take photo of camera view for reference
- Measure distances if possible
- Test by doing full motion - ensure everything visible

### Object Position Variation

**For 100 Episodes - Recommended Distribution:**

**Strategy 1: Grid Pattern**
- Mark 10 positions in a grid across workspace
- Record 10 episodes per position
- Positions should be 2-3 inches apart

**Strategy 2: Systematic Coverage**
- 5 positions in front area (near robot)
- 5 positions in back area (far from robot)
- 10 episodes per position
- Include variety of distances and angles

**Key Principles:**
- Cover the full workspace area you want the robot to handle
- More variation = better generalization
- But don't make positions too random - robot needs to see patterns

### Object Rotation (For Better Generalization)

**If collecting 100+ episodes:**
- At each position, rotate object: 0°, 45°, 90°, 135°, 180°
- Example: Position 1 with 5 rotations = 5 episodes
- 10 positions × 10 rotations each = 100 episodes

**Start simple:**
- First 100 episodes: varied positions, same rotation
- Next 100 episodes: add rotation variation

### Target Location (Box/Bowl)

**For initial training:**
- **Keep target FIXED** in exact same position
- Robot will learn to place at that location
- Moving target during training = much harder task

**For advanced training:**
- After achieving 80%+ success with fixed target
- Record new dataset with 3-5 target positions
- 20 episodes per target position
- Train on combined dataset

### Demonstration Quality

**Each episode should:**
- Complete the task successfully (pick object → place in target)
- Use smooth, consistent motions
- Take 10-20 seconds (not too slow, not rushed)
- Start and end in similar arm configurations
- Show clear grasp and release actions

**Consistency is key:**
- Grasp object the same way each time (or with systematic variation)
- Use similar motion speed across episodes
- Follow same general trajectory
- Release cleanly into target

**Common mistakes to avoid:**
- Bumping objects accidentally
- Arm collisions with workspace
- Incomplete grasps (object falling)
- Placing object outside target
- Very slow or very fast movements

### Lighting

**Requirements:**
- Consistent lighting across all episodes
- Avoid strong shadows
- Natural or artificial light is fine
- **Critical:** Lighting during deployment must match training

**Recommendation:**
- Use consistent indoor lighting
- Close blinds if using natural light (varies by time of day)
- Add desk lamps if needed for consistency

### How Much Data?

**Minimum viable:**
- 50 episodes: Can work for very simple, consistent tasks
- Success rate: 60-80%

**Recommended:**
- 100-150 episodes: Good generalization for varied positions
- Success rate: 80-90%

**Advanced:**
- 200-300 episodes: Robust performance with object/target variation
- Success rate: 90-95%

**More data benefits:**
- Better generalization
- More robust to position variation
- Higher success rates
- But diminishing returns after ~300 episodes

---

## Training

### Training Command
```bash
export HF_HUB_OFFLINE=1

lerobot-train \
    --policy.type=act \
    --dataset.repo_id=local/my_dataset \
    --steps=50000 \
    --batch_size=8 \
    --eval_freq=5000 \
    --output_dir=./outputs/my_training \
    --policy.repo_id=local/my_policy \
    --policy.push_to_hub=false \
    --policy.device=cuda
```

### Training Parameters Explained

**--steps:** Total training steps
- 20,000: Quick test (1-2 hours on RTX 3060)
- 50,000: Recommended (2-4 hours on RTX 3060)
- 100,000: Advanced (4-8 hours on RTX 3060)

**--batch_size:** Samples per training step
- Larger = faster training but more GPU memory
- RTX 3060 (12GB): Use 8
- Smaller GPU: Try 4

**--eval_freq:** How often to save checkpoints
- 5,000: Saves at steps 5k, 10k, 15k, etc.
- Can resume from any checkpoint if training interrupted

### Monitoring Training

**Watch for:**
- Loss decreasing steadily (good sign)
- Loss plateaus or increases (may need more data or different parameters)
- GPU utilization ~90-100% (efficient training)
- Time per step consistent (~0.15-0.20 seconds on RTX 3060)

**Good training signs:**
- Loss starts high (5-10) and drops to <0.2
- Gradient norms decrease over time
- No NaN or Inf values

### Training Output
```bash
# Model saved to:
./outputs/my_training/checkpoints/050000/pretrained_model/

# Or use "last" symlink:
./outputs/my_training/checkpoints/last/pretrained_model/
```

### Training Time Estimates (RTX 3060)
- 50 episodes, 20K steps: ~1-1.5 hours
- 100 episodes, 50K steps: ~2.5-4 hours
- 200 episodes, 100K steps: ~5-8 hours

---

## Deployment

### Testing Trained Policy
```bash
export HF_HUB_OFFLINE=1

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm1 \
    --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: 327122076093, width: 640, height: 480, fps: 30}}" \
    --display_data=true \
    --dataset.repo_id=local/eval_test \
    --dataset.single_task="Evaluating policy" \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=30 \
    --dataset.push_to_hub=false \
    --policy.path=./outputs/my_training/checkpoints/050000/pretrained_model
```

### Deployment Checklist

**Before running:**
- [ ] Camera in EXACT same position as training
- [ ] Same lighting conditions
- [ ] Object and target in workspace
- [ ] Clear space around robot
- [ ] Power switch accessible for emergency stop

**During first test:**
- Watch closely for unexpected movements
- Be ready to cut power immediately
- Start with object in a trained position
- Manually reset between episodes

### Troubleshooting Poor Performance

**Robot misses object:**
- Camera position different from training
- Object position outside training distribution
- Need more training data at that position

**Robot moves erratically:**
- Calibration issue - recalibrate
- Model undertrained - train more steps
- Check for hardware issues

**Success rate varies by position:**
- Normal! Training data distribution matters
- Positions seen more in training = better performance
- Solution: Add more data at problematic positions

### Expected Performance

**50 episodes, 20K steps:**
- 40-60% success rate at trained positions
- Poor generalization to new positions

**100 episodes, 50K steps:**
- 70-85% success rate at trained positions
- Moderate generalization to nearby positions

**200 episodes, 50K+ steps:**
- 85-95% success rate at trained positions
- Good generalization across workspace

---

### Common Issues

**Issue: Follower arm not responding**
- Solution: Check USB permissions, power cycle arm

**Issue: Camera not found**
- Solution: Check USB connection, verify serial number with lerobot-find-cameras

**Issue: Training loss not decreasing**
- Solution: Check data quality, increase training steps, collect more data

**Issue: Robot works in some positions but not others**
- Solution: Need more training data at problematic positions

**Issue: Success rate drops over time**
- Solution: Camera or lighting changed - verify setup

---

## Quick Reference Commands

### Setup
```bash
# Check devices
ls -l /dev/ttyACM*

# Fix permissions
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1

# Find cameras
lerobot-find-cameras
```

### Calibration
```bash
# Leader
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=leader_arm1

# Follower
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower_arm1
```

### Teleoperation
```bash
lerobot-teleoperate \
    --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=follower_arm1 \
    --teleop.type=so101_leader --teleop.port=/dev/ttyACM0 --teleop.id=leader_arm1 \
    --robot.cameras="{ front: {type: intelrealsense, serial_number_or_name: SERIAL, width: 640, height: 480, fps: 30}}" \
    --display_data=true
```

### Data Collection
```bash
export HF_HUB_OFFLINE=1
lerobot-record [same params as teleoperate] \
    --dataset.repo_id=local/my_dataset \
    --dataset.single_task="Task description" \
    --dataset.fps=30 \
    --dataset.num_episodes=100 \
    --dataset.episode_time_s=25 \
    --dataset.reset_time_s=10 \
    --dataset.push_to_hub=false
```

### Training
```bash
export HF_HUB_OFFLINE=1
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=local/my_dataset \
    --steps=50000 \
    --batch_size=8 \
    --output_dir=./outputs/my_training \
    --policy.repo_id=local/my_policy \
    --policy.push_to_hub=false \
    --policy.device=cuda
```

### Deployment
```bash
export HF_HUB_OFFLINE=1
lerobot-record [robot and camera params] \
    --dataset.repo_id=local/eval \
    --dataset.single_task="Eval" \
    --dataset.num_episodes=10 \
    --dataset.push_to_hub=false \
    --policy.path=./outputs/my_training/checkpoints/last/pretrained_model
```

---
