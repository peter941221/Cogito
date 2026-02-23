"""Tests for world/renderer.py."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from numpy.random import Generator

from cogito.config import Config
from cogito.world.grid import CogitoWorld
from cogito.world.renderer import WorldRenderer, BatchRenderer


@pytest.fixture
def rng() -> Generator:
    """Seeded random generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def world(rng: Generator) -> CogitoWorld:
    """Create a test world."""
    return CogitoWorld(Config, rng)


class TestWorldRenderer:
    """Tests for WorldRenderer."""

    def test_create_headless(self):
        """Renderer creates successfully in headless mode."""
        renderer = WorldRenderer(Config, headless=True)
        assert renderer is not None
        renderer.close()

    def test_create_with_display(self):
        """Renderer creates successfully with display."""
        renderer = WorldRenderer(Config, headless=False)
        assert renderer is not None
        renderer.close()

    def test_render_updates_image(self, world: CogitoWorld):
        """render() updates the image data."""
        with WorldRenderer(Config, headless=True) as renderer:
            renderer.render(world, (32, 32), 100.0, 0, force=True)
            assert renderer.image_data is not None
            assert renderer.image_data.shape == (64, 64, 3)

    def test_render_respects_interval(self, world: CogitoWorld):
        """render() respects render_interval setting."""
        with WorldRenderer(Config, headless=True, render_interval=100) as renderer:
            # First render
            renderer.render(world, (32, 32), 100.0, 0, force=True)
            assert renderer.last_rendered_step == 0

            # Should not render at step 50
            renderer.render(world, (32, 32), 100.0, 50)
            assert renderer.last_rendered_step == 0

            # Should render at step 100
            renderer.render(world, (32, 32), 100.0, 100)
            assert renderer.last_rendered_step == 100

    def test_render_forced(self, world: CogitoWorld):
        """render() with force=True ignores interval."""
        with WorldRenderer(Config, headless=True, render_interval=100) as renderer:
            renderer.render(world, (32, 32), 100.0, 0, force=True)
            renderer.render(world, (32, 32), 100.0, 10, force=True)
            assert renderer.last_rendered_step == 10

    def test_save_frame_creates_file(self, world: CogitoWorld):
        """save_frame() creates a valid image file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "test_frame.png"

            with WorldRenderer(Config, headless=True) as renderer:
                renderer.render(world, (32, 32), 100.0, 0, force=True)
                renderer.save_frame(str(filename))

            assert filename.exists()
            assert filename.stat().st_size > 0

    def test_save_frame_creates_directory(self, world: CogitoWorld):
        """save_frame() creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "nested" / "dir" / "test.png"

            with WorldRenderer(Config, headless=True) as renderer:
                renderer.render(world, (32, 32), 100.0, 0, force=True)
                renderer.save_frame(str(filename))

            assert filename.exists()

    def test_context_manager(self):
        """Renderer works as context manager."""
        with WorldRenderer(Config, headless=True) as renderer:
            assert renderer.fig is not None
        # Figure should be closed after exit

    def test_agent_marker_visible(self, world: CogitoWorld):
        """Agent marker is created on render."""
        with WorldRenderer(Config, headless=True) as renderer:
            renderer.render(world, (10, 20), 100.0, 0, force=True)
            assert renderer.agent_marker is not None


class TestBatchRenderer:
    """Tests for BatchRenderer."""

    def test_create(self):
        """BatchRenderer creates successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with BatchRenderer(Config, output_dir=tmpdir) as renderer:
                assert renderer is not None

    def test_save_at_interval(self, world: CogitoWorld):
        """render_and_save() saves at correct intervals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with BatchRenderer(Config, output_dir=tmpdir, save_interval=10) as renderer:
                # Should not save at step 5
                result = renderer.render_and_save(world, (32, 32), 100.0, 5)
                assert result is None

                # Should save at step 10
                result = renderer.render_and_save(world, (32, 32), 100.0, 10)
                assert result is not None
                assert Path(result).exists()

    def test_output_directory_created(self):
        """Output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new" / "dir"
            with BatchRenderer(Config, output_dir=str(output_dir)):
                pass
            assert output_dir.exists()

    def test_frame_naming(self, world: CogitoWorld):
        """Frame files are named sequentially."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with BatchRenderer(Config, output_dir=tmpdir, save_interval=1) as renderer:
                path1 = renderer.render_and_save(world, (32, 32), 100.0, 1)
                path2 = renderer.render_and_save(world, (32, 32), 100.0, 2)

            assert "frame_000000" in str(path1)
            assert "frame_000001" in str(path2)
