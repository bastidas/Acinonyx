

from PIL import Image
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import svgwrite
import argparse

def run_clicks_on_gif(svg_filename: str):
    """
    Function to handle clicks on a GIF and save the clicked points as SVG and CSV.
    """
    pass  # Placeholder for the function, actual implementation below
    clicked_points = []
    # Load the GIF
    base_dir = 'movement_reference'
    gif_path = os.path.join(base_dir, 'running.gif')
    gif = Image.open(gif_path)

    # Function to handle clicks on the frame
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            current_points.append((event.xdata, event.ydata))
            if len(current_points) == 2:
                plt.close()  # Close the current frame after two clicks

    # Iterate through frames
    i = 0
    while True:
        current_points = []
        plt.imshow(gif)
        plt.title(f"Frame {gif.tell()} (Click two points for arrow)")
        plt.gcf().canvas.mpl_connect('button_press_event', on_click)
        plt.show(block=True)  # Block execution until the plot window is closed

        if len(current_points) == 2:
            clicked_points.append(tuple(current_points))
        else:
            # If user closes window without two clicks, break
            break

        try:
            gif.seek(gif.tell() + 1)  # Move to the next frame
        except EOFError:
            break  # Exit the loop when no more frames are available  


    # Print the collected points
    print("Clicked points (pairs):", clicked_points)
    csv_filename = f"{os.path.splitext(gif_path)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x1', 'y1', 'x2', 'y2'])
        for pair in clicked_points:
            (x1, y1), (x2, y2) = pair
            writer.writerow([x1, y1, x2, y2])

    print(f"Saved clicked point pairs to {csv_filename}")

    #svg_filename = f"{os.path.splitext(gif_path)[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg"
    svg_filename = os.path.join(base_dir, svg_filename)
    width, height = gif.size
    dwg = svgwrite.Drawing(svg_filename, size=(width, height))
    dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill='none'))  # Transparent background
    color = "white"
    color = "white"
    arrow_marker = dwg.marker(id='arrow', insert=(2, 2), size=(4, 4), orient='auto')
    arrow_marker.add(
        dwg.path(d="M0,0 L4,2 L0,4 L1,2 Z", fill=color)
    )
    dwg.defs.add(arrow_marker)

    # Draw a grey line connecting the second point in each clicked_points
    if len(clicked_points) > 1:
        second_points = [pair[1] for pair in clicked_points]
        dwg.add(
            dwg.polyline(points=second_points, stroke='grey', stroke_width=1, fill='none')
        )

    for idx, ((x1, y1), (x2, y2)) in enumerate(clicked_points):
        # Draw start and end points
        dwg.add(dwg.circle(center=(x1, y1), r=3, fill='red'))
        dwg.add(dwg.circle(center=(x2, y2), r=3, fill='orange'))
        # Draw vector (arrow) from (x1, y1) to (x2, y2)
        dwg.add(dwg.line(start=(x1, y1),
                        end=(x2, y2),
                        stroke=color,
                        stroke_width=2,
                        marker_end=arrow_marker.get_funciri()))
        # Optionally, label the vector
        dwg.add(dwg.text(str(idx + 1), insert=(x2 + 5, y2 - 5), font_size='12px', fill=color))

    dwg.save()
    print(f"Saved clicked point pairs as SVG to {svg_filename}")

    #print(f"Saved clicked points as SVG to {svg_filename}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process GIF and save clicked points as SVG and CSV.")
    parser.add_argument('-f', '--filename', type=str, help='Filename to use for the SVG and CSV output (without extension)')
    args = parser.parse_args()

    if args.filename:
        base_filename = args.filename
        svg_filename = f"{base_filename}.svg"
        csv_filename = f"{base_filename}.csv"
    run_clicks_on_gif(svg_filename)

    


