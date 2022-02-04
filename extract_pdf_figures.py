#!/usr/bin/env python3
"""
PDF Figure Extraction Script
Extracts all figures from a PDF and saves them as PNG files with custom names.
"""

import fitz  # PyMuPDF
import os
from pathlib import Path

def extract_figures_from_pdf(pdf_path, output_dir):
    """
    Extract all figures from a PDF file and save them as PNG files.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the extracted figures
    """

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Figure mapping
    figure_mapping = {
        0: "implementation_architecture.png",
        1: "system_architecture_detailed.png",
        2: "config_screen.png",
        3: "controls_config.png",
        4: "first_screen.png",
        5: "track1.png",
        6: "track2.png",
        7: "dataset_sample.png",
        8: "driving_log_csv.png",
        9: "accompanied_model.png",
        10: "nvidia_architecture.png",
        11: "crop_image.png",
        12: "flip_image.png",
        13: "shift_vertical.png",
        14: "shift_horizontal.png",
        15: "brightness.png",
        16: "random_shadows.png",
        17: "random_blur.png",
        18: "loss_epochs.png",
    }

    # Open the PDF
    pdf_document = fitz.open(pdf_path)

    figure_count = 0
    extracted_count = 0

    # Iterate through all pages
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]

        # Get all images on the page
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            pix = fitz.Pixmap(pdf_document, xref)

            # Convert to RGB if necessary
            if pix.n - pix.alpha < 4:  # RGB
                pix_rgb = pix
            else:  # CMYK
                pix_rgb = fitz.Pixmap(fitz.csRGB, pix)

            # Determine output filename
            if figure_count in figure_mapping:
                output_filename = figure_mapping[figure_count]
                output_path = os.path.join(output_dir, output_filename)

                # Save as PNG with high quality
                pix_rgb.save(output_path)
                print(f"✓ Extracted Figure {figure_count + 1}: {output_filename}")
                extracted_count += 1
            else:
                print(f"⚠ Skipped Figure {figure_count + 1} (no mapping defined)")

            figure_count += 1

            # Clean up
            if pix_rgb != pix:
                pix_rgb = None
            pix = None

    pdf_document.close()

    print(f"\n{'='*60}")
    print(f"Extraction complete!")
    print(f"Total figures found: {figure_count}")
    print(f"Total figures extracted: {extracted_count}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")

    return extracted_count


def main():
    pdf_path = "/Users/kunalbhujbal/Desktop/RP/Self Driving Car/2023028317 (Paper PDF).pdf"
    output_dir = "/Users/kunalbhujbal/Desktop/Projects/AutoPilot/docs"

    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return

    print(f"Starting PDF figure extraction...")
    print(f"PDF Path: {pdf_path}")
    print(f"Output Directory: {output_dir}")
    print(f"{'='*60}\n")

    extract_figures_from_pdf(pdf_path, output_dir)


if __name__ == "__main__":
    main()
