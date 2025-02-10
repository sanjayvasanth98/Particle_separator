# Particle Separator

## Description  
This repository contains the **Particle Separator** process, designed to enhance **Particle Image Velocimetry (PIV) preprocessing and post-processing**. The code detects particles using a **single-point maxima/minima algorithm**, ensuring accurate localization of individual particles. The detected particles are then **overlaid on a black background**, making them suitable for **PIV processing**.

Additionally, the repository includes **post-processing scripts** for analyzing **PIV results**, enabling further insights into flow characteristics.

## Features  
- **Particle Detection:** Implements a **single-point maxima/minima** approach for accurate identification.  
- **Preprocessing for PIV:** Converts detected particles into a **high-contrast black background image** for input into PIV algorithms.  
- **Post-Processing:** Scripts for analyzing, visualizing, and interpreting PIV results.  

## Usage  
1. **Run the particle detection script** to identify and overlay particles.  
2. **Feed the generated images** into a **PIV solver** for velocity field calculation.  
3. **Utilize post-processing scripts** to analyze and extract meaningful **flow data**.  

This repository is optimized for **experimental fluid mechanics applications**, particularly **cavitation and particle tracking studies**. 
