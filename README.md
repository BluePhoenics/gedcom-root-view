# GEDCOM Family Tree Visualization

A specialized tool for visualizing deep family trees (10+ generations) with optimized space management for pedigree collapse and missing ancestors. Perfect for genealogists working with complex ancestry data where traditional fan charts fall short.

## Key Features ✨
- **Deep Generation Support**: Designed for trees with 10+ generations (tested up to 13)
- **Pedigree Collapse Handling**: Intelligent angle allocation for duplicate ancestors
- **Missing Ancestor Optimization**: Compact layout that doesn't waste space on unknown branches
- **Radial Visualization**: Fan chart-inspired layout with Bézier curve connections
- **GEDCOM Ready**: Works with standard genealogy files (.ged)
- **Dynamic Styling**: Color-coded by gender with marriage date annotations

## Installation 📦
```bash
pip install matplotlib numpy ged4py

## Requirements
This script requires the following Python libraries:
- `matplotlib`
- `ged4py`
- `numpy`

You can install missing dependencies using:
```
pip install matplotlib ged4py numpy
```

## Usage
1. Place your GEDCOM file in the project folder
2. Modify the following variables at the beginning of the script:
   - `gedcom_file`: Path to the GEDCOM file.
   - `main_family_id`: Root family ID.
   - `max_gen`: Maximum generations to display.
   - `output_figsize`: 
3. Run the script:
   ```
   python gedcom-root-view.py
   ```

## Customization
Parameter	Description
max_gen	Controls how many generations to display (reduce for denser trees)
max_radius_step	Adjust spacing between generations (2.0 = compact, 3.0 = more spaced)
dist_bend	Controls connection curve tightness (0.3-1.5)
line_width	Connection line thickness

## Output
The script generates a visualization displaying family relationships, where:
- Males are in blue, females in red.
- Family nodes (marriages) are in green.
- Connections are represented with smooth curved lines.

## License
This project is licensed under the MIT License.
