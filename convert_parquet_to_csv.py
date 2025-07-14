#!/usr/bin/env python3
"""
Convert MNIST parquet files with PNG images to pixel arrays for C++ loading
"""
import pyarrow.parquet as pq
import numpy as np
import os
from PIL import Image
import io

def convert_parquet_to_pixels():
    """Convert MNIST parquet files with PNG images to pixel CSV format"""
    
    # File paths
    data_dir = "data/mnist"
    train_parquet = os.path.join(data_dir, "train.parquet")
    test_parquet = os.path.join(data_dir, "test.parquet")
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    
    print("Converting MNIST PNG images to pixel arrays...")
    
    def decode_png_to_pixels(png_bytes):
        """Decode PNG bytes to 28x28 pixel array"""
        try:
            # Open PNG from bytes
            image = Image.open(io.BytesIO(png_bytes))
            
            # Convert to grayscale and resize to 28x28 if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            if image.size != (28, 28):
                image = image.resize((28, 28), Image.LANCZOS)
            
            # Convert to numpy array and flatten
            pixels = np.array(image).flatten()
            
            return pixels
        except Exception as e:
            print(f"Error decoding PNG: {e}")
            return None
    
    def convert_file(parquet_path, csv_path, file_type):
        if not os.path.exists(parquet_path):
            print(f"Error: {parquet_path} not found")
            return
            
        print(f"\nLoading {parquet_path}...")
        
        # Read parquet file
        table = pq.read_table(parquet_path)
        print(f"{file_type} data shape: {table.num_rows} rows, {table.num_columns} columns")
        
        # Convert to python objects
        data_dict = table.to_pydict()
        images = data_dict['image']
        labels = data_dict['label']
        
        print(f"Converting {len(images)} PNG images to pixel arrays...")
        
        # Open CSV file for writing
        with open(csv_path, 'w') as f:
            # Write header: label,pixel0,pixel1,...,pixel783
            header = "label," + ",".join([f"pixel{i}" for i in range(784)])
            f.write(header + "\n")
            
            successful_conversions = 0
            
            for i, (image_data, label) in enumerate(zip(images, labels)):
                # Extract PNG bytes
                png_bytes = image_data['bytes']
                
                # Decode PNG to pixels
                pixels = decode_png_to_pixels(png_bytes)
                
                if pixels is not None and len(pixels) == 784:
                    # Write row: label,pixel0,pixel1,...,pixel783
                    row = f"{label}," + ",".join([str(p) for p in pixels])
                    f.write(row + "\n")
                    successful_conversions += 1
                else:
                    print(f"Warning: Failed to decode image {i}")
                
                # Progress indicator
                if (i + 1) % 5000 == 0:
                    print(f"  Processed {i + 1}/{len(images)} images...")
            
            print(f"{file_type} conversion complete: {successful_conversions}/{len(images)} images")
        
        # Show file size
        size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print(f"{file_type} CSV size: {size_mb:.1f} MB")
        
        # Verify first few pixels of first image
        if successful_conversions > 0:
            print(f"Sample {file_type} data verification:")
            with open(csv_path, 'r') as f:
                f.readline()  # Skip header
                first_line = f.readline().strip()
                parts = first_line.split(',')
                label = parts[0]
                first_10_pixels = parts[1:11]
                print(f"  First image: label={label}, first 10 pixels: {first_10_pixels}")
    
    # Convert both files
    convert_file(train_parquet, train_csv, "Training")
    convert_file(test_parquet, test_csv, "Test")
    
    print("\nConversion complete! MNIST PNG images converted to pixel arrays.")
    print("Files ready for C++ loading:")
    print(f"  {data_dir}/train.csv")
    print(f"  {data_dir}/test.csv")

if __name__ == "__main__":
    try:
        convert_parquet_to_pixels()
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nInstall required packages:")
        print("pip install pyarrow pillow")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
