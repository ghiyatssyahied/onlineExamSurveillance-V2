#!/usr/bin/env python3
"""
Centralized configuration management for Cheating Surveillance System
"""
import json
import os
from datetime import datetime

class CheatingDetectionConfig:
    """Centralized configuration manager for all detection systems"""
    
    def __init__(self, config_file="detection_config.json"):
        self.config_file = config_file
        self.config = self.load_default_config()
        self.load_config()
    
    def load_default_config(self):
        """Load default configuration parameters"""
        return {
            # Eye Movement Detection
            "eye_detection": {
                "horizontal_threshold": 0.12,
                "vertical_threshold": 0.10,
                "high_confidence_threshold": 0.8,
                "min_contrast": 30,
                "min_eye_region_size": 144,
                "history_size": 7,
                "confidence_history_size": 5,
                "majority_vote_threshold": 2,
                "pupil_area_min": 20,
                "pupil_circularity_min": 0.3
            },
            
            # Head Pose Detection
            "head_pose": {
                "yaw_threshold": 12,
                "pitch_threshold_up": 10,
                "pitch_threshold_down": 12,
                "roll_threshold": 8,
                "center_yaw_tolerance": 7,
                "center_pitch_tolerance": 7,
                "center_roll_tolerance": 6,
                "angle_history_size": 15,
                "calibration_frames": 20,
                "state_change_delay": 0.2,
                "outlier_threshold": 30,
                "focal_length_multiplier": 1.2,
                "min_contrast_threshold": 30
            },
            
            # Mobile Detection
            "mobile_detection": {
                "confidence_threshold": 0.7,
                "model_path": "best_yolov12.pt",
                "mobile_class_index": 0,
                "use_gpu": True
            },
            
            # System Settings
            "system": {
                "camera_index": 1,
                "fallback_camera_index": 0,
                "log_directory": "log",
                "screenshot_delay": 3.0,        # Seconds to wait before taking screenshot
                "mobile_screenshot_delay": 2.0,  # Shorter delay for mobile detection
                "max_screenshots_per_type": 100, # Limit screenshots to prevent disk overflow
                "debug_mode": False
            },
            
            # Performance Settings
            "performance": {
                "max_fps": 30,
                "frame_skip": 1,
                "enable_gpu_acceleration": True,
                "memory_limit_mb": 1024
            },
            
            # Accuracy Tuning
            "tuning": {
                "auto_tune_enabled": True,
                "sensitivity_mode": "balanced",  # low, balanced, high
                "calibration_quality_threshold": 0.8,
                "minimum_detection_confidence": 0.5
            }
        }
    
    def load_config(self):
        """Load configuration from file if exists"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge with defaults (file config takes precedence)
                self._deep_merge(self.config, file_config)
                print(f"Configuration loaded from {self.config_file}")
                
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
        else:
            print("Using default configuration (no config file found)")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            # Add metadata
            config_with_meta = {
                "metadata": {
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat(),
                    "description": "Cheating Detection System Configuration"
                },
                **self.config
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_with_meta, f, indent=4)
            
            print(f"✅ Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving config file: {e}")
            return False
    
    def get(self, section, key=None):
        """Get configuration value"""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def set(self, section, key, value):
        """Set configuration value"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        
        # Auto-save if enabled
        if self.get("system", "auto_save_config"):
            self.save_config()
    
    def update_section(self, section, updates):
        """Update entire section with new values"""
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section].update(updates)
        print(f"Updated {section} configuration: {updates}")
    
    def get_sensitivity_preset(self, mode="balanced"):
        """Get sensitivity preset configurations"""
        presets = {
            "low": {
                # Less sensitive - fewer false positives
                "eye_detection": {
                    "horizontal_threshold": 0.15,
                    "vertical_threshold": 0.12,
                    "high_confidence_threshold": 0.9
                },
                "head_pose": {
                    "yaw_threshold": 15,
                    "pitch_threshold_up": 12,
                    "pitch_threshold_down": 15,
                    "center_yaw_tolerance": 10,
                    "center_pitch_tolerance": 10
                }
            },
            "balanced": {
                # Current optimized settings
                "eye_detection": {
                    "horizontal_threshold": 0.12,
                    "vertical_threshold": 0.10,
                    "high_confidence_threshold": 0.8
                },
                "head_pose": {
                    "yaw_threshold": 12,
                    "pitch_threshold_up": 10,
                    "pitch_threshold_down": 12,
                    "center_yaw_tolerance": 7,
                    "center_pitch_tolerance": 7
                }
            },
            "high": {
                # More sensitive - may have more false positives
                "eye_detection": {
                    "horizontal_threshold": 0.08,
                    "vertical_threshold": 0.08,
                    "high_confidence_threshold": 0.6
                },
                "head_pose": {
                    "yaw_threshold": 8,
                    "pitch_threshold_up": 6,
                    "pitch_threshold_down": 8,
                    "center_yaw_tolerance": 5,
                    "center_pitch_tolerance": 5
                }
            }
        }
        
        return presets.get(mode, presets["balanced"])
    
    def apply_sensitivity_preset(self, mode="balanced"):
        """Apply sensitivity preset to configuration"""
        preset = self.get_sensitivity_preset(mode)
        
        for section, settings in preset.items():
            self.update_section(section, settings)
        
        self.set("tuning", "sensitivity_mode", mode)
        print(f"Applied sensitivity preset: {mode}")
    
    def _deep_merge(self, base_dict, override_dict):
        """Deep merge two dictionaries"""
        for key, value in override_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def print_config(self):
        """Print current configuration in a readable format"""
        print("\n" + "="*50)
        print("CHEATING DETECTION SYSTEM CONFIGURATION")
        print("="*50)
        
        for section, settings in self.config.items():
            print(f"\n[{section.upper()}]")
            for key, value in settings.items():
                print(f"  {key}: {value}")
        
        print("="*50 + "\n")
    
    def validate_config(self):
        """Validate configuration parameters"""
        issues = []
        
        # Validate eye detection thresholds
        eye_config = self.get("eye_detection")
        if eye_config.get("horizontal_threshold", 0) <= 0:
            issues.append("Eye horizontal_threshold must be > 0")
        
        # Validate head pose thresholds  
        head_config = self.get("head_pose")
        if head_config.get("yaw_threshold", 0) <= 0:
            issues.append("Head yaw_threshold must be > 0")
        
        # Validate system settings
        system_config = self.get("system")
        if not os.path.exists(system_config.get("log_directory", "log")):
            issues.append(f"Log directory does not exist: {system_config.get('log_directory')}")
        
        if issues:
            print("⚠️  Configuration validation issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ Configuration validation passed")
            return True

# Global configuration instance
config = CheatingDetectionConfig()

def get_config():
    """Get global configuration instance"""
    return config

def reload_config():
    """Reload configuration from file"""
    global config
    config = CheatingDetectionConfig()
    return config

if __name__ == "__main__":
    # Demo/test configuration system
    config = CheatingDetectionConfig()
    config.print_config()
    config.validate_config()
    
    # Test sensitivity presets
    print("\nTesting sensitivity presets:")
    for mode in ["low", "balanced", "high"]:
        print(f"\n{mode.upper()} sensitivity:")
        preset = config.get_sensitivity_preset(mode)
        for section, settings in preset.items():
            print(f"  {section}: {settings}")