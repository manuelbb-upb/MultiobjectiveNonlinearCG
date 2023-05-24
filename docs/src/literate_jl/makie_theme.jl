import Makie: Theme, Contourf
using LaTeXStrings

DOC_DPI = 300

DOC_SIZE_CM = (10, 9)
DOC_SIZE_INCHES = DOC_SIZE_CM ./ 2.54
DOC_RESOLUTION = ceil.(Int, DOC_SIZE_INCHES .* DOC_DPI)

DOC_THEME = Theme(
    resolution = DOC_RESOLUTION,
    markersize = 40f0,
    fontsize = 38f0,
    linewidth = 4f0,
    colormap=:acton,
    Label = (
        fontsize=42f0,
    ),
    Axis = (
        xlabel = L"x_1",
        ylabel = L"x_2",
        xlabelsize = 50f0,
        ylabelsize = 50f0,
    ),
)

DOC_SIZE_CM2 = (10, 5.5)
DOC_SIZE_INCHES2 = DOC_SIZE_CM2 ./ 2.54
DOC_RESOLUTION2 = ceil.(Int, DOC_SIZE_INCHES2 .* DOC_DPI)

DOC_THEME2 = Theme(
    resolution = DOC_RESOLUTION2,
    markersize = 25f0,
    fontsize = 28f0,
    linewidth = 2.5f0,
    colormap=:acton,
    Label = (
        fontsize=33f0,
    ),
    Axis = (
        xlabel = L"x_1",
        ylabel = L"x_2",
        xlabelsize = 40f0,
        ylabelsize = 40f0,
    ),
)