from infinigen.datagen.manage_jobs import init_db_from_existing
from tu.loggers.utils import print_viscam_url, print_vcv_url
from pathlib import Path
from engine.constants import PROJ_DIR
from jinja2 import Environment, FileSystemLoader, select_autoescape


def make_html_page(output_path, scenes, **kwargs):
    env = Environment(
        loader=FileSystemLoader(Path(PROJ_DIR) / 'engine/utils'),
        autoescape=select_autoescape(),
    )

    template = env.get_template("infinigen_template.html")
    seeds = [scene['seed'] for scene in scenes]
    html = template.render(
        seeds=seeds,
        frames=list(range(1, 5)),
        **kwargs,
    )

    with output_path.open('w') as f:
        f.write(html)


def write_html_summary(output_folder: str, max_size=5000, overwrite: bool = False):
    output_folder = Path(output_folder)
    all_scenes = init_db_from_existing(output_folder)
    all_scenes = list(filter(None, all_scenes))
    names = [
        "index" if (idx == 0) else f"index_{idx}"
        for idx in range(0, len(all_scenes), max_size)
    ]
    for name, idx in zip(names, range(0, len(all_scenes), max_size)):
        html_path = output_folder / f"{name}.html"
        if overwrite or not html_path.exists():
            make_html_page(html_path, all_scenes[idx:idx + max_size], pages=names)

            print(print_viscam_url(html_path))
