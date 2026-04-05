import numpy as np
import pandas as pd
import os

# ===============================
# GENERATE TIME-SERIES TACTILE DATA
# ===============================
def generate_tomato_properties(class_name):
    """Generate physical properties for Roma tomato"""
    # Length L_i (5-7.6 cm)
    L_i = np.random.uniform(0.05, 0.076)  # meters
    
    # Diameter D_i (3.5-4.5 cm)  
    D_i = np.random.uniform(0.035, 0.045)  # meters
    
    # Class-specific properties
    if class_name == 'Ripe':
        weight_g = np.random.uniform(60, 76)
        f_i = np.random.uniform(6.8, 10.0)  # Softer
    else:  # occluded
        weight_g = np.random.uniform(23, 70)
        f_i = np.random.uniform(10.0, 14.0)  # Firmer
    
    m_i = weight_g / 1000  # kg
    a_i = D_i / 2
    c_i = L_i / 2
    V_i = (4/3) * np.pi * a_i**2 * c_i
    A_i = 0.25 * np.pi * (0.4 * D_i) * (0.3 * L_i)
    r_i = 0.6 * D_i
    
    return {
        'm_i_kg': m_i,
        'A_i_m2': A_i,
        'r_i_m': r_i,
        'f_i_N': f_i,
        'class_name': class_name,
        'L_i_m': L_i,
        'D_i_m': D_i,
        'weight_g': weight_g
    }

def generate_time_series_tactile(props, time_steps=20):
    """Convert static physics to time-series data"""
    m_i = props['m_i_kg']
    A_i = props['A_i_m2']
    r_i = props['r_i_m']
    f_i = props['f_i_N']
    class_name = props['class_name']
    
    t = np.linspace(0, 1, time_steps)
    time_series_data = []
    
    for step in range(time_steps):
        time_factor = t[step]
        
        # Force evolution
        if time_factor < 0.3:
            grip_profile = 0.5 + 1.5 * (time_factor/0.3)
        elif time_factor < 0.8:
            grip_profile = 2.0 + 0.2 * np.sin(5*time_factor)
        else:
            grip_profile = 2.2 - 0.2 * ((time_factor-0.8)/0.2)
        
        F_i_t = m_i * 9.81 * grip_profile
        
        # Pressure evolution
        deformation = 1.0 - 0.3 * (1 - np.exp(-5*time_factor))
        A_i_t = A_i * deformation
        P_i_t = F_i_t / A_i_t
        
        # Torque evolution
        if class_name == 'Ripe':
            slip_factor = 1.0 - 0.2 * np.sin(3*time_factor)
        else:
            slip_factor = 1.0 - 0.1 * np.sin(2*time_factor)
        
        τ_i_t = F_i_t * r_i * slip_factor
        
        # Firmness response
        if class_name == 'Ripe':
            deformation_response = 0.8 + 0.4 * (1 - np.exp(-3*time_factor))
        else:
            deformation_response = 0.6 + 0.2 * (1 - np.exp(-2*time_factor))
        
        f_i_t = f_i * deformation_response
        
        # Time to pluck
        T_i_t = 1.0 * (m_i * f_i_t) / (F_i_t + 1e-6)
        
        # Vibration signal
        if time_factor > 0.4:
            vibration = 0.05 * np.sin(20*time_factor) * np.exp(-2*(time_factor-0.4))
        else:
            vibration = 0
        
        # 10 features per time step
        time_step_features = [
            F_i_t, P_i_t, τ_i_t, f_i_t, T_i_t,
            vibration, time_factor, grip_profile,
            deformation, slip_factor
        ]
        
        time_series_data.append(time_step_features)
    
    return np.array(time_series_data)

def generate_lstm_tactile_dataset():
    """Main function to generate time-series tactile dataset"""
    # Your image counts
    image_counts = {
        'train': {'occluded': 3839, 'Ripe': 3330},
        'val': {'occluded': 3409, 'Ripe': 1739},
        'test': {'occluded': 3422, 'Ripe': 1734}
    }
    
    output_dir = r"C:\..PhD Thesis\DataSet\Tactile_TimeSeries"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING TIME-SERIES TACTILE DATA")
    print("=" * 60)
    
    time_steps = 20
    features_per_step = 10
    
    for split in ['train', 'val', 'test']:
        print(f"\n📁 Generating {split} split...")
        
        all_time_series = []
        all_labels = []
        all_metadata = []
        
        for class_name in ['occluded', 'Ripe']:
            num_samples = image_counts[split][class_name]
            print(f"  {class_name}: {num_samples} samples")
            
            for i in range(num_samples):
                # Generate properties
                tomato_id = f"{split}_{class_name}_{i+1:05d}"
                props = generate_tomato_properties(class_name)
                props['tomato_id'] = tomato_id
                
                # Generate time-series
                time_series = generate_time_series_tactile(props, time_steps)
                
                # Store
                all_time_series.append(time_series)
                all_labels.append(0 if class_name == 'Ripe' else 1)
                
                # Metadata
                metadata = {
                    'tomato_id': tomato_id,
                    'split': split,
                    'label': class_name,
                    'L_i_m': props['L_i_m'],
                    'D_i_m': props['D_i_m'],
                    'weight_g': props['weight_g'],
                    'f_i_N': props['f_i_N']
                }
                all_metadata.append(metadata)
        
        # Save as numpy arrays
        X = np.array(all_time_series)
        y = np.array(all_labels)
        
        np.save(os.path.join(output_dir, f"X_{split}_tactile.npy"), X)
        np.save(os.path.join(output_dir, f"y_{split}_tactile.npy"), y)
        
        # Save metadata
        metadata_df = pd.DataFrame(all_metadata)
        metadata_df.to_csv(os.path.join(output_dir, f"metadata_{split}.csv"), index=False)
        
        print(f"  ✅ Saved {X.shape} samples")
        print(f"     Shape: {X.shape[0]} × {X.shape[1]} × {X.shape[2]}")
    
    print("\n" + "=" * 60)
    print("TACTILE TIME-SERIES DATASET GENERATED")
    print("=" * 60)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Each sample: 20 time steps × 10 features")
    print(f"🎯 Ready for LSTM feature extraction")

# ===============================
# RUN THE GENERATOR
# ===============================
if __name__ == "__main__":
    generate_lstm_tactile_dataset()