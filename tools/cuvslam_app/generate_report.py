# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import os
import json
import base64
import subprocess
from datetime import datetime

from jinja2 import Environment, FileSystemLoader


def save_stats_to_json(stats, output_dir):
    """Save all stats to a single JSON file."""

    stats_dir = os.path.join(output_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)

    # Convert stats to list of dictionaries
    stats_list = []
    for stat in stats:
        stat_dict = {
            'sequence_title': stat.sequence_title,
            'n_frames': stat.n_frames,
            'tracking_time': stat.tracking_time,
            'average_fps': stat.average_fps,
            'bird_view_with_errors_path': stat.bird_view_with_errors_path,
            'gt_av_translation_error': stat.gt_av_translation_error,
            'gt_av_rotation_error': stat.gt_av_rotation_error,
            'gt_n_error_segments': stat.gt_n_error_segments,
            'gt_simple_error': stat.gt_simple_error,
            'num_tracking_losts': stat.num_tracking_losts,
            'odometry_mode': stat.odometry_mode
        }
        stats_list.append(stat_dict)

    # Save to JSON file
    json_file = os.path.join(stats_dir, "all_stats.json")
    with open(json_file, 'w') as f:
        json.dump(stats_list, f, indent=2)

    print(f"Saved statistics for {len(stats_list)} sequences to {json_file}")


def get_fps(tracking_time, n_frames):
    if tracking_time > 0:
        fps = n_frames / tracking_time
        return fps
    else:
        return -1


def image_to_base64(image_path):
    """Convert an image file to base64 data URI."""
    if not os.path.exists(image_path):
        return ""

    try:
        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            base64_data = base64.b64encode(img_data).decode('utf-8')
            # Detect image format from extension
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = 'image/png' if ext == '.png' else 'image/jpeg'
            return f"data:{mime_type};base64,{base64_data}"
    except Exception as e:
        print(f"Error converting image to base64: {image_path}, {e}")
        return ""


def filter_rows(stats, suffix):
    return list(filter(lambda x: x['sequence_title'].endswith(suffix), stats))


def calc_summary(title, stats):
    tracking_time = 0
    n_frames = 0
    total_gt_av_translation_error = 0
    total_gt_av_rotation_error = 0
    total_gt_n_error_segments = 0
    total_gt_simple_error = 0
    num_correct_sequences = 0
    num_stats_with_gt = 0
    total_tracking_losts = 0
    for s in stats:
        tracking_time += s.tracking_time
        n_frames += s.n_frames
        total_gt_av_translation_error += s.gt_av_translation_error
        total_gt_av_rotation_error += s.gt_av_rotation_error
        total_gt_n_error_segments += s.gt_n_error_segments
        total_gt_simple_error += s.gt_simple_error
        total_tracking_losts += s.num_tracking_losts
        if s.gt_av_translation_error > 0:
            num_stats_with_gt += 1
            if s.gt_av_translation_error <= 0.011:
                num_correct_sequences += 1

    summary_av_gt_translation_error = 0
    summary_av_gt_rotation_error = 0

    summary_av_gt_simple_error = 0
    if num_stats_with_gt > 0:
        summary_av_gt_translation_error = total_gt_av_translation_error / num_stats_with_gt
        summary_av_gt_rotation_error = total_gt_av_rotation_error / num_stats_with_gt
        summary_av_gt_simple_error = total_gt_simple_error / num_stats_with_gt
    return {
        "title": title,
        "n_frames": n_frames,
        "average_fps": get_fps(tracking_time, n_frames),
        "summary_av_gt_translation_error": summary_av_gt_translation_error,
        "summary_av_gt_rotation_error": summary_av_gt_rotation_error,
        "summary_av_gt_simple_error": summary_av_gt_simple_error,
        "total_sequences": num_stats_with_gt,
        "num_correct_sequences": num_correct_sequences,
        "total_tracking_losts": total_tracking_losts
    }


def generate_report(test_folder, comments, stats, generate_pdf=False, config_name=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get commit SHA
    try:
        commit_sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            universal_newlines=True,
            cwd=script_dir
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_sha = "unknown"

    # Get branch name
    try:
        branch_name = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            universal_newlines=True,
            cwd=script_dir
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        branch_name = "unknown"

    # Get commit timestamp
    try:
        commit_ts = subprocess.check_output(
            ['git', 'show', '-s', '--format=%ci', commit_sha],
            universal_newlines=True,
            cwd=script_dir
        ).strip().replace(' ', '/')
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_ts = "unknown"

    date_time = datetime.now().replace(microsecond=0).astimezone().isoformat('/')

    os.makedirs(test_folder, exist_ok=True)

    # Use config name if provided, otherwise use default name
    report_basename = f"report_{config_name}" if config_name else "report"

    # For HTML: reference images in existing plots folder
    stats_for_html = []
    for s in stats:
        # Calculate relative path from report to existing plot image
        image_relative_path = ""
        if s.bird_view_with_errors_path and os.path.exists(s.bird_view_with_errors_path):
            # Get relative path from test_folder to the image
            try:
                image_relative_path = os.path.relpath(s.bird_view_with_errors_path, test_folder)
            except ValueError:
                # If on different drives (Windows), use absolute path
                image_relative_path = s.bird_view_with_errors_path

        stat_dict = {
            'sequence_title': s.sequence_title,
            'n_frames': s.n_frames,
            'average_fps': s.average_fps,
            'gt_av_translation_error': s.gt_av_translation_error,
            'gt_av_rotation_error': s.gt_av_rotation_error,
            'gt_simple_error': s.gt_simple_error,
            'bird_view_with_errors_path': s.bird_view_with_errors_path,
            'bird_view_image_path': image_relative_path  # Relative path for HTML
        }
        stats_for_html.append(stat_dict)

    template_dir = os.path.join(os.path.dirname(__file__), 'report_templates')
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

    # Generate HTML with image references
    template = env.get_template("report.html")
    total = calc_summary("total", stats)
    # TODO: provide correct units of measurements for use_segments=false, %, deg
    html = template.render(
        date=date_time,
        branch_name=branch_name,
        commit_sha=commit_sha,
        commit_ts=commit_ts,
        comments=comments,
        stats=stats_for_html,
        total=total
    )

    html_file_name = os.path.join(test_folder, f"{report_basename}.html")
    with open(html_file_name, "wt") as f:
        f.write(html)

    print(f"{html_file_name} was done")

    # Generate PDF if requested (with embedded base64 images)
    if generate_pdf:
        try:
            from weasyprint import HTML

            # For PDF: convert images to base64 for embedding
            stats_for_pdf = []
            for s in stats:
                stat_dict = {
                    'sequence_title': s.sequence_title,
                    'n_frames': s.n_frames,
                    'average_fps': s.average_fps,
                    'gt_av_translation_error': s.gt_av_translation_error,
                    'gt_av_rotation_error': s.gt_av_rotation_error,
                    'gt_simple_error': s.gt_simple_error,
                    'bird_view_with_errors_path': s.bird_view_with_errors_path,
                    'bird_view_base64': image_to_base64(s.bird_view_with_errors_path) if s.bird_view_with_errors_path else ""
                }
                stats_for_pdf.append(stat_dict)

            # Use PDF-specific template
            pdf_template = env.get_template("report_pdf.html")
            pdf_html = pdf_template.render(
                date=date_time,
                branch_name=branch_name,
                commit_sha=commit_sha,
                commit_ts=commit_ts,
                comments=comments,
                stats=stats_for_pdf,
                total=total
            )

            pdf_file_name = os.path.join(test_folder, f"{report_basename}.pdf")
            HTML(string=pdf_html).write_pdf(pdf_file_name)
            print(f"{pdf_file_name} was done")
        except ImportError:
            print("Warning: weasyprint not installed. Install with 'pip install weasyprint' to generate PDF reports.")
        except Exception as e:
            print(f"Error generating PDF: {e}")
