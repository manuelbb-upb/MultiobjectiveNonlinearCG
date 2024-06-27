import CairoMakie: Makie, Theme, Contourf
using LaTeXStrings

DOC_DPI = 300

DOC_SIZE_CM = (10, 9)
DOC_SIZE_INCHES = DOC_SIZE_CM ./ 2.54
DOC_RESOLUTION = ceil.(Int, DOC_SIZE_INCHES .* DOC_DPI)

DOC_THEME = Theme(
    size = DOC_RESOLUTION,
    markersize = 40f0,
    fontsize = 38f0,
    linewidth = 6f0,
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
    size = DOC_RESOLUTION2,
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

WONG_COLORS = Makie.wong_colors()
DOC_COLORS = Dict(
    :PS => WONG_COLORS[3], 
    :PF => WONG_COLORS[3],
    :min => WONG_COLORS[5],
    :sd => WONG_COLORS[6], 
    :sdSZ => WONG_COLORS[6], 
    :sdM => WONG_COLORS[7], 
    :prp3 => WONG_COLORS[4],
    :prp3SZ => WONG_COLORS[4],
    :prpOrth => WONG_COLORS[1], 
    :frRestart => WONG_COLORS[2],
)
DOC_LSTYLES = Dict(
    :PS => :solid,
    :PF => :solid,
    :sd => :solid,
    :sdSZ => :dash,
    :sdM => :solid,
    :prp3 => :solid,
    :prp3SZ => :dash,
    :prpOrth => :solid,
    :frRestart => :solid,
)