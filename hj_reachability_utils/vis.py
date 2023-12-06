""" Utility functions for visualization.
"""
import numpy as np

def vis_2d_level_set(fig, ax, grid, values, level=0.0, colormap=True, contour_color=None):
        vxs = grid.coordinate_vectors[0]
        vys = grid.coordinate_vectors[1]
        meshgrid_table_vy, meshgrid_table_vx = np.meshgrid(vys, vxs)
        if colormap:
            p = ax.pcolor(meshgrid_table_vx, meshgrid_table_vy, values, cmap='RdBu')
            fig.colorbar(p, ax=ax)
        if contour_color is None:
            contour_color = 'w' if colormap else 'b'
        s2 = ax.contour(meshgrid_table_vx, meshgrid_table_vy, values, 
                            level * np.ones(1), colors=contour_color,
                            linestyles='solid', origin='lower', linewidths=1.0)