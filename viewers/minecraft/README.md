# Running the offline Minecraft Renderer

A Flask application to serve `.litematic` files, commonly used for Minecraft schematics.

## Installation

### Prerequisites

- Python 3.7+
- pip
- flask

There's not a lot of requirements, I tried to write it as barebones as possible - no React, no Typescript, no Node...so it should be runnable in most environments. You just need flask.

## Running the Application

Start the Flask app:

```bash
python run.py
```

The home page will give you either the option to select a loaded `.litematic` file from your local, or drag and drop a new one. Press Home at any given top (top left corner) to go back. Use WASD to control the direction, your mouse drag to pan, and SHIFT and SPACE to go up/down.

## Adding schematics

Add schematics to `app/static/schematics`. Depending on if it's from an experiment or it's downloaded, add it to exps or static respectively.
