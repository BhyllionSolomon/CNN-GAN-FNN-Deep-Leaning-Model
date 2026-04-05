"""
Tactile Data Generator for Roma Tomatoes
Generates physical properties based on RIPENESS only (not occlusion)
Occlusion is visual only - does not change physical properties
"""

import numpy as np
import pandas as pd
import os

# ===============================
# PHYSICAL CONSTANTS
# ===============================
GRAVITY = 9.81  # m/s²

# ===============================
# GENERATE TOMATO PHYSICAL PROPERTIES
# ===============================
def generate_tomato_properties(ripeness):
    """
    Generate physical properties for Roma tomato based on RIPENESS only.
    Occlusion does NOT affect physical properties.
    
    Args:
        ripeness: 'Ripe' or 'Unripe'
    
    Returns:
        Dictionary of physical properties
    """
    # Length L_i (5.0 - 7.6 cm) - consistent for all tomatoes
    L_i = np.random.uniform(0.050, 0.076)  # meters
    
    # Diameter D_i (3.5 - 4.5 cm) - consistent for all tomatoes
    D_i = np.random.uniform(0.035, 0.045)  # meters
    
    # Class-specific properties based on RIPENESS only
    if ripeness == 'Ripe':
        # Ripe tomatoes: heavier, softer
        weight_g = np.random.uniform(60, 76)        # grams
        firmness_N = np.random.uniform(6.8, 10.0)   # Newtons (softer)
    else:  # Unripe
        # Unripe tomatoes: lighter, firmer
        weight_g = np.random.uniform(23, 45)        # grams (unripe are smaller/lighter)
        firmness_N = np.random.uniform(10.0, 14.0)  # Newtons (firmer)
    
    # Derived properties
    mass_kg = weight_g / 1000  # kg
    
    # Volume (ellipsoid: V = 4/3 * π * a² * c, where a = D/2, c = L/2)
    a = D_i / 2
    c = L_i / 2
    volume_m3 = (4/3) * np.pi * a**2 * c
    
    # Contact area (simplified - elliptical contact)
    contact_area_m2 = 0.25 * np.pi * (0.4 * D_i) * (0.3 * L_i)
    
    # Lever arm for torque (distance from gripper center to fruit center)
    lever_arm_m = 0.6 * D_i
    
    # Density
    density_kg_m3 = mass_kg / volume_m3 if volume_m3 > 0 else 900
    
    return {
        'ripeness': ripeness,
        'length_m': L_i,
        'diameter_m': D_i,
        'weight_g': weight_g,
        'mass_kg': mass_kg,
        'volume_m3': volume_m3,
        'density_kg_m3': density_kg_m3,
        'contact_area_m2': contact_area_m2,
        'lever_arm_m': lever_arm_m,
        'firmness_N': firmness_N
    }

# ===============================
# GENERATE TIME-SERIES SENSOR READINGS
# ===============================
def generate_grip_time_series(props, time_steps=20):
    """
    Simulate time-series sensor readings during a grip event.
    
    Args:
        props: Physical properties from generate_tomato_properties()
        time_steps: Number of time steps to generate
    
    Returns:
        Array of shape (time_steps, 10) with sensor readings
    """
    m = props['mass_kg']
    A = props['contact_area_m2']
    r = props['lever_arm_m']
    f = props['firmness_N']
    ripeness = props['ripeness']
    
    # Time vector (0 to 1 second grip duration)
    t = np.linspace(0, 1.0, time_steps)
    
    # Calculate weight (force due to gravity)
    weight_N = m * GRAVITY
    
    time_series = []
    
    for step, time_factor in enumerate(t):
        # ===== 1. GRIP FORCE EVOLUTION =====
        # Grip force increases, holds, then releases
        if time_factor < 0.3:  # Approach and initial contact (0-0.3s)
            force_factor = 0.5 + 1.5 * (time_factor / 0.3)
        elif time_factor < 0.8:  # Stable grip (0.3-0.8s)
            # Small oscillations from sensor noise
            force_factor = 2.0 + 0.05 * np.sin(20 * time_factor)
        else:  # Release (0.8-1.0s)
            force_factor = 2.0 * (1.0 - (time_factor - 0.8) / 0.2)
        
        # FIXED: Grip force should be greater than weight to securely hold the tomato
        # Minimum grip force = 1.5x weight, maximum = 4.5x weight
        grip_force_N = weight_N * (1.5 + force_factor)
        
        # ===== 2. CONTACT PRESSURE =====
        # Pressure = Force / Area (with slight deformation)
        deformation = 1.0 - 0.2 * (1 - np.exp(-3 * time_factor))
        current_area = A * deformation
        pressure_Pa = grip_force_N / current_area
        
        # ===== 3. TORQUE =====
        # Torque depends on grip force and fruit orientation
        if ripeness == 'Ripe':
            # Ripe tomatoes may slip slightly
            slip_factor = 1.0 - 0.1 * np.sin(5 * time_factor)
        else:
            # Unripe tomatoes grip better
            slip_factor = 1.0 - 0.05 * np.sin(3 * time_factor)
        
        torque_Nm = grip_force_N * r * slip_factor
        
        # ===== 4. FIRMNESS RESPONSE =====
        # Firmness affects deformation rate
        if ripeness == 'Ripe':
            # Ripe deforms more
            deformation_response = 0.7 + 0.5 * (1 - np.exp(-4 * time_factor))
        else:
            # Unripe deforms less
            deformation_response = 0.5 + 0.3 * (1 - np.exp(-3 * time_factor))
        
        firmness_response_N = f * deformation_response
        
        # ===== 5. TIME TO PLUCK (estimated) =====
        # Time to pluck = (mass * firmness) / grip force
        time_to_pluck_s = 0.8 * (m * firmness_response_N) / (grip_force_N + 1e-6)
        
        # ===== 6. VIBRATION SIGNAL =====
        # Vibrations during grip (from motor, contact)
        if time_factor > 0.3 and time_factor < 0.85:
            # Vibrations during stable grip
            vibration = 0.03 * np.sin(30 * time_factor) * (1 + 0.2 * np.sin(5 * time_factor))
        else:
            vibration = 0.0
        
        # ===== 7. GRIP VELOCITY =====
        # Rate of force change (derivative)
        if step > 0:
            prev_force = time_series[-1][0] if time_series else grip_force_N
            grip_velocity = (grip_force_N - prev_force) / (t[1] - t[0])
        else:
            grip_velocity = 0.0
        
        # ===== 8. TEMPERATURE (slight increase during grip) =====
        temperature_C = 22.0 + 0.5 * time_factor
        
        # ===== 9. SLIP RISK =====
        # Risk of slipping (higher for ripe, during movement)
        if ripeness == 'Ripe':
            slip_risk = 0.2 + 0.3 * abs(grip_velocity / 50) + 0.1 * np.sin(8 * time_factor)
        else:
            slip_risk = 0.1 + 0.2 * abs(grip_velocity / 50) + 0.05 * np.sin(5 * time_factor)
        slip_risk = min(1.0, max(0.0, slip_risk))
        
        # ===== 10. STABILITY SCORE =====
        # Overall grip stability (inverse of slip risk)
        stability_score = 1.0 - slip_risk
        
        # Collect all 10 features for this time step
        step_features = [
            grip_force_N,           # 1. Grip force (N)
            pressure_Pa,            # 2. Contact pressure (Pa)
            torque_Nm,              # 3. Torque (Nm)
            firmness_response_N,    # 4. Dynamic firmness (N)
            time_to_pluck_s,        # 5. Estimated pluck time (s)
            vibration,              # 6. Vibration amplitude
            grip_velocity,          # 7. Rate of force change (N/s)
            temperature_C,          # 8. Temperature (°C)
            slip_risk,              # 9. Slip risk (0-1)
            stability_score         # 10. Stability score (0-1)
        ]
        
        time_series.append(step_features)
    
    return np.array(time_series, dtype=np.float32)

# ===============================
# GENERATE COMPLETE DATASET
# ===============================
def generate_tactile_dataset():
    """
    Main function to generate tactile dataset matching image counts.
    Physical properties are based on RIPENESS only (not occlusion).
    """
    # Image counts from your dataset
    # IMPORTANT: These counts are for IMAGES, not physical tomatoes
    # An occluded image shows the same physical tomato as a visible one
    image_counts = {
        'train': {
            'Ripe': 3330,        # Fully visible ripe tomatoes
            'Unripe': 3839        # Occluded tomatoes (but physically unripe)
        },
        'val': {
            'Ripe': 1739,         # Fully visible ripe tomatoes
            'Unripe': 3409         # Occluded tomatoes (but physically unripe)
        },
        'test': {
            'Ripe': 1734,         # Fully visible ripe tomatoes
            'Unripe': 3422         # Occluded tomatoes (but physically unripe)
        }
    }
    
    output_dir = r"C:\..PhD Thesis\DataSet\Tactile_Readings"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING TACTILE DATASET FOR ROMA TOMATOES")
    print("=" * 70)
    print("\n📊 Physical properties based on RIPENESS only:")
    print("   • Ripe tomatoes: heavier (60-76g), softer (6.8-10.0N)")
    print("   • Unripe tomatoes: lighter (23-45g), firmer (10.0-14.0N)")
    print("   • Occlusion is VISUAL only - does not affect physical properties\n")
    
    time_steps = 20  # 20 time steps per grip event
    
    for split in ['train', 'val', 'test']:
        print(f"\n📁 Generating {split.upper()} split...")
        
        all_time_series = []
        all_labels = []          # 0=Ripe, 1=Unripe (physical ripeness)
        all_occlusion_labels = [] # 0=Visible, 1=Occluded (visual only)
        all_metadata = []
        
        # Generate for Ripe tomatoes (visible in images)
        num_ripe = image_counts[split]['Ripe']
        print(f"  🍅 Ripe (visible): {num_ripe} samples")
        
        for i in range(num_ripe):
            # Generate physical properties for RIPE tomato
            props = generate_tomato_properties('Ripe')
            props['sample_id'] = f"{split}_ripe_{i+1:05d}"
            props['visual_status'] = 'visible'
            
            # Generate time-series sensor readings
            time_series = generate_grip_time_series(props, time_steps)
            
            all_time_series.append(time_series)
            all_labels.append(0)  # 0 = Ripe
            all_occlusion_labels.append(0)  # 0 = Visible
            
            # Store metadata
            metadata = {
                'sample_id': props['sample_id'],
                'split': split,
                'ripeness': 'Ripe',
                'visual_status': 'visible',
                'weight_g': props['weight_g'],
                'firmness_N': props['firmness_N'],
                'length_m': props['length_m'],
                'diameter_m': props['diameter_m'],
                'density_kg_m3': props['density_kg_m3']
            }
            all_metadata.append(metadata)
        
        # Generate for Unripe tomatoes (occluded in images)
        num_unripe = image_counts[split]['Unripe']
        print(f"  🍅 Unripe (occluded): {num_unripe} samples")
        
        for i in range(num_unripe):
            # Generate physical properties for UNRIPE tomato
            props = generate_tomato_properties('Unripe')
            props['sample_id'] = f"{split}_unripe_{i+1:05d}"
            props['visual_status'] = 'occluded'
            
            # Generate time-series sensor readings (SAME physical properties as visible unripe)
            time_series = generate_grip_time_series(props, time_steps)
            
            all_time_series.append(time_series)
            all_labels.append(1)  # 1 = Unripe
            all_occlusion_labels.append(1)  # 1 = Occluded (visually)
            
            # Store metadata
            metadata = {
                'sample_id': props['sample_id'],
                'split': split,
                'ripeness': 'Unripe',
                'visual_status': 'occluded',
                'weight_g': props['weight_g'],
                'firmness_N': props['firmness_N'],
                'length_m': props['length_m'],
                'diameter_m': props['diameter_m'],
                'density_kg_m3': props['density_kg_m3']
            }
            all_metadata.append(metadata)
        
        # Convert to numpy arrays
        X = np.array(all_time_series, dtype=np.float32)
        y = np.array(all_labels, dtype=np.int32)
        y_occlusion = np.array(all_occlusion_labels, dtype=np.int32)
        
        # Verify counts match
        total_expected = image_counts[split]['Ripe'] + image_counts[split]['Unripe']
        print(f"  ✅ Generated {X.shape[0]} samples (expected {total_expected})")
        print(f"     Shape: {X.shape[0]} samples × {X.shape[1]} time steps × {X.shape[2]} features")
        print(f"     Ripe: {np.sum(y == 0)}, Unripe: {np.sum(y == 1)}")
        print(f"     Visible: {np.sum(y_occlusion == 0)}, Occluded: {np.sum(y_occlusion == 1)}")
        
        # Save files
        np.save(os.path.join(output_dir, f"X_{split}_tactile.npy"), X)
        np.save(os.path.join(output_dir, f"y_{split}_ripeness.npy"), y)
        np.save(os.path.join(output_dir, f"y_{split}_occlusion.npy"), y_occlusion)
        
        # Save metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(os.path.join(output_dir, f"metadata_{split}.csv"), index=False)
    
    print("\n" + "=" * 70)
    print("✅ TACTILE DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n📁 Output directory: {output_dir}")
    print("\n📊 Files saved:")
    print("   • X_train_tactile.npy - Time-series sensor data")
    print("   • y_train_ripeness.npy - Physical ripeness labels (0=Ripe, 1=Unripe)")
    print("   • y_train_occlusion.npy - Visual occlusion labels (0=Visible, 1=Occluded)")
    print("   • metadata_train.csv - Sample metadata")
    print("\n🎯 Next step: Extract features using LSTM for grip force prediction")

# ===============================
# RUN THE GENERATOR
# ===============================
if __name__ == "__main__":
    generate_tactile_dataset()