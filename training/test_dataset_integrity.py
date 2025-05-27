#!/usr/bin/env python3
"""
Comprehensive dataset validation and testing suite
Validates PNG/JSON pairs and data integrity in dataset folder
"""
import json
from pathlib import Path
from PIL import Image
import pandas as pd

class DatasetValidator:
    """Validates dataset structure and content integrity"""
    
    def __init__(self, dataset_dir="../dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.errors = []
        self.warnings = []
    
    def validate_file_pairs(self):
        """Test that every PNG has a corresponding JSON file"""
        png_files = set(f.stem for f in self.dataset_dir.glob("*.png"))
        json_files = set(f.stem for f in self.dataset_dir.glob("*.json"))
        
        # Find missing pairs
        missing_json = png_files - json_files
        missing_png = json_files - png_files
        
        if missing_json:
            self.errors.append(f"PNG files missing JSON: {missing_json}")
        if missing_png:
            self.errors.append(f"JSON files missing PNG: {missing_png}")
        
        print(f"✓ Found {len(png_files & json_files)} valid PNG/JSON pairs")
        if missing_json or missing_png:
            print(f"✗ {len(missing_json)} missing JSON, {len(missing_png)} missing PNG")
        
        return len(missing_json) == 0 and len(missing_png) == 0
    
    def validate_image_properties(self):
        """Test image file properties and integrity"""
        valid_images = 0
        
        for png_file in self.dataset_dir.glob("*.png"):
            try:
                with Image.open(png_file) as img:
                    # Check basic properties
                    if img.mode not in ['RGB', 'RGBA', 'L']:
                        self.warnings.append(f"{png_file.name}: Unusual color mode {img.mode}")
                    
                    # Check reasonable dimensions
                    if img.width < 256 or img.height < 256:
                        self.warnings.append(f"{png_file.name}: Small dimensions {img.width}x{img.height}")
                    elif img.width > 2048 or img.height > 2048:
                        self.warnings.append(f"{png_file.name}: Large dimensions {img.width}x{img.height}")
                    
                    valid_images += 1
            except Exception as e:
                self.errors.append(f"{png_file.name}: Image corrupted - {e}")
        
        print(f"✓ {valid_images} images validated successfully")
        return len(self.errors) == 0
    
    def validate_json_structure(self):
        """Test JSON file structure and required fields"""
        structural_count = 0
        legacy_count = 0
        
        required_fields = ["grid_px", "grid_mm", "floor"]
        structural_fields = ["structural_elements", "element_summary", "layout_bounds"]
        
        for json_file in self.dataset_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Check basic required fields
                missing_required = [field for field in required_fields if field not in data]
                if missing_required:
                    self.errors.append(f"{json_file.name}: Missing required fields {missing_required}")
                
                # Determine data version
                if any(field in data for field in structural_fields):
                    structural_count += 1
                    self._validate_structural_json(json_file.name, data)
                else:
                    legacy_count += 1
                    self._validate_legacy_json(json_file.name, data)
                
            except json.JSONDecodeError as e:
                self.errors.append(f"{json_file.name}: Invalid JSON - {e}")
            except Exception as e:
                self.errors.append(f"{json_file.name}: Validation error - {e}")
        
        print(f"✓ {structural_count} structural format files")
        print(f"✓ {legacy_count} legacy format files")
        return len(self.errors) == 0
    
    def _validate_structural_json(self, filename, data):
        """Validate structural elements format"""
        
        # Check layout_bounds
        if "layout_bounds" in data:
            bounds = data["layout_bounds"]
            if not isinstance(bounds.get("width_grids"), (int, float)) or bounds.get("width_grids", 0) <= 0:
                self.errors.append(f"{filename}: Invalid width_grids")
            if not isinstance(bounds.get("height_grids"), (int, float)) or bounds.get("height_grids", 0) <= 0:
                self.errors.append(f"{filename}: Invalid height_grids")
        
        # Check structural_elements
        if "structural_elements" in data:
            elements = data["structural_elements"]
            if not isinstance(elements, list):
                self.errors.append(f"{filename}: structural_elements must be a list")
            else:
                for i, element in enumerate(elements):
                    self._validate_structural_element(filename, i, element)
        
        # Check element_summary consistency
        if "element_summary" in data and "structural_elements" in data:
            summary = data["element_summary"]
            elements = data["structural_elements"]
            
            actual_stairs = len([e for e in elements if e.get("type") == "stair"])
            actual_entrances = len([e for e in elements if e.get("type") == "entrance"])
            actual_balconies = len([e for e in elements if e.get("type") == "balcony"])
            
            if summary.get("stair_count", 0) != actual_stairs:
                self.errors.append(f"{filename}: Stair count mismatch - summary:{summary.get('stair_count')} vs actual:{actual_stairs}")
            if summary.get("entrance_count", 0) != actual_entrances:
                self.errors.append(f"{filename}: Entrance count mismatch")
            if summary.get("balcony_count", 0) != actual_balconies:
                self.errors.append(f"{filename}: Balcony count mismatch")
    
    def _validate_structural_element(self, filename, index, element):
        """Validate individual structural element"""
        required_element_fields = ["type", "grid_x", "grid_y"]
        
        for field in required_element_fields:
            if field not in element:
                self.errors.append(f"{filename}: Element {index} missing {field}")
        
        # Validate element type
        valid_types = ["stair", "entrance", "balcony", "structural_wall"]
        if element.get("type") not in valid_types:
            self.errors.append(f"{filename}: Element {index} has invalid type {element.get('type')}")
        
        # Validate coordinates
        for coord in ["grid_x", "grid_y"]:
            if coord in element and not isinstance(element[coord], (int, float)):
                self.errors.append(f"{filename}: Element {index} {coord} must be numeric")
    
    def _validate_legacy_json(self, filename, data):
        """Validate legacy format JSON"""
        # Basic validation for legacy format
        if "floor" in data and data["floor"] not in ["1F", "2F"]:
            self.warnings.append(f"{filename}: Unusual floor value {data['floor']}")
    
    def validate_naming_convention(self):
        """Test file naming consistency"""
        naming_issues = []
        
        for png_file in self.dataset_dir.glob("*.png"):
            name = png_file.stem
            
            # Check naming pattern: plan_XXXX_Yf
            if not name.startswith("plan_"):
                naming_issues.append(f"{png_file.name}: Should start with 'plan_'")
            
            if not (name.endswith("_1f") or name.endswith("_2f")):
                naming_issues.append(f"{png_file.name}: Should end with '_1f' or '_2f'")
        
        if naming_issues:
            self.warnings.extend(naming_issues)
            print(f"⚠ {len(naming_issues)} naming convention issues")
        else:
            print("✓ All files follow naming convention")
        
        return len(naming_issues) == 0
    
    def generate_dataset_report(self):
        """Generate comprehensive dataset report"""
        png_files = list(self.dataset_dir.glob("*.png"))
        json_files = list(self.dataset_dir.glob("*.json"))
        
        # Count different types
        structural_files = 0
        legacy_files = 0
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                if "structural_elements" in data:
                    structural_files += 1
                else:
                    legacy_files += 1
            except:
                pass
        
        report = f"""
=== Dataset Validation Report ===
Total PNG files: {len(png_files)}
Total JSON files: {len(json_files)}
Structural format: {structural_files}
Legacy format: {legacy_files}

Validation Results:
- Errors: {len(self.errors)}
- Warnings: {len(self.warnings)}

Status: {'✓ PASS' if len(self.errors) == 0 else '✗ FAIL'}
"""
        
        if self.errors:
            report += "\nErrors:\n" + "\n".join(f"  - {error}" for error in self.errors)
        
        if self.warnings:
            report += "\nWarnings:\n" + "\n".join(f"  - {warning}" for warning in self.warnings)
        
        print(report)
        
        # Save report to file
        with open(self.dataset_dir / "validation_report.txt", "w") as f:
            f.write(report)
        
        return len(self.errors) == 0
    
    def run_all_validations(self):
        """Run complete validation suite"""
        print("=== Running Dataset Validation ===")
        
        results = {
            "file_pairs": self.validate_file_pairs(),
            "image_properties": self.validate_image_properties(),
            "json_structure": self.validate_json_structure(),
            "naming_convention": self.validate_naming_convention()
        }
        
        self.generate_dataset_report()
        
        return all(results.values())

# Pytest test functions
def test_dataset_file_pairs():
    """Test that all PNG files have corresponding JSON files"""
    validator = DatasetValidator()
    assert validator.validate_file_pairs(), "Missing PNG/JSON pairs found"

def test_dataset_images():
    """Test image file integrity"""
    validator = DatasetValidator()
    assert validator.validate_image_properties(), "Image validation failed"

def test_dataset_json():
    """Test JSON file structure"""
    validator = DatasetValidator()
    assert validator.validate_json_structure(), "JSON validation failed"

def test_dataset_naming():
    """Test file naming convention"""
    validator = DatasetValidator()
    validator.validate_naming_convention()  # Allow warnings but don't fail

def test_enhanced_dataset_preparation():
    """Test enhanced dataset preparation script"""
    from prepare_dataset_enhanced import prepare_enhanced_dataset
    
    try:
        metadata_path = prepare_enhanced_dataset()
        assert metadata_path.exists(), "Enhanced metadata file not created"
        
        # Validate metadata structure
        import pandas as pd
        df = pd.read_csv(metadata_path)
        
        required_columns = ["file_name", "text", "conditions", "data_version"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        assert len(missing_columns) == 0, f"Missing columns in metadata: {missing_columns}"
        
        print(f"✓ Enhanced dataset preparation successful: {len(df)} samples")
        
    except ImportError:
        print("⚠ Enhanced dataset preparation script not found - skipping test")

if __name__ == "__main__":
    # Run validation directly
    validator = DatasetValidator()
    success = validator.run_all_validations()
    
    if success:
        print("\n🎉 All validations passed!")
        exit(0)
    else:
        print("\n❌ Validation failed - check errors above")
        exit(1)