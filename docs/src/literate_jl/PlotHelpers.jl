module PlotHelpers

    using Colors
    import Colors: RGBA
    import MultiobjectiveNonlinearCG: AbstractStepRule, SteepestDescentDirection,
        FletcherReevesRestart, FletcherReevesFractionalLP,
        PRP3, PRPConeProjection

    const upbUltraBlue = colorant"#0025AA"
    const upbFuchsiaRed = colorant"#C138A0"
    const upbSkyBlue = colorant"#0A75C4"
    const upbSaphireBlue = colorant"#181C62"
    const upbOceanBlue = colorant"#23AFC9"
    const upbIrisViolet = colorant"#7E3FA8"
    const upbArcticBlue = colorant"#50D1D1"

    const upbMediumGray = colorant"#5F5F5F"
    const upbPomegranatePink = colorant"#EF3A84"
    const upbLimeGreen = colorant"#acea3d"


    const UPB_COLORS = Base.ImmutableDict(
        :ultra_blue => upbUltraBlue,
        :fuchsia_red => upbFuchsiaRed,
        :sky_blue => upbSkyBlue,
        :saphire_blue => upbSaphireBlue,
        :ocean_blue => upbOceanBlue,
        :iris_violet => upbIrisViolet,
        :arctic_blue => upbArcticBlue,
        :medium_gray => upbMediumGray,
        :pomegranate_pink => upbPomegranatePink,
        :lime_green => upbLimeGreen,
    )

    upb_palette = [
        upbUltraBlue, upbFuchsiaRed, upbLimeGreen, upbPomegranatePink, upbOceanBlue, upbIrisViolet,  
    ]
    pset_color() = upbArcticBlue

    step_rule_color(::AbstractStepRule) = upbUltraBlue
    step_rule_color(::SteepestDescentDirection) = upbUltraBlue
    step_rule_color(::FletcherReevesRestart) = upbLimeGreen
    step_rule_color(::FletcherReevesFractionalLP) = upbIrisViolet
    step_rule_color(::PRP3) = upbSkyBlue
    step_rule_color(::PRPConeProjection) = upbPomegranatePink

    step_rule_label(::AbstractStepRule) = ""
    step_rule_label(::SteepestDescentDirection) = "SD"
    step_rule_label(::FletcherReevesRestart) = "FR_⏯"
    step_rule_label(::FletcherReevesFractionalLP) = "FR_÷"
    step_rule_label(::PRP3) = "PRP_3"
    step_rule_label(::PRPConeProjection) = "PRP_⟂"

    import CairoMakie: RGB, RGBA
    function darker_color(base_color::RGB{F}; factor=.5) where F
        return RGB{F}(
            base_color.r * factor,
            base_color.g * factor,
            base_color.b * factor,
        )
    end
    function darker_color(base_color::RGBA{F}; kwargs...) where F
        dark_rgb = darker_color(RGB(base_color))
        return RGBA{F}(dark_rgb, base_color.alpha)
    end
    function cmap_gradient(base_color; kwargs...)
        dark_color = darker_color(base_color; kwargs...)
        return [base_color, dark_color]
    end

    function critval_markersizes(
        critvals, transform=log, T=Float32;
        min_sz = 10, factor = min_sz
    )
        cvs = T.(transform.(critvals))
        cv_max = maximum(cvs)
        cvs ./= cv_max
        cvs .*= factor
        cvs .+= min_sz
        return cvs
    end
   
    function _get_limits(Xs; margin_type=:rel, margins=(.02, .02))
        xlims, ylims = mapreduce(box_limits, envelop, Xs)
        if isa(margins, Real)
            margins = (margins, margins)
        end
        abs_margins = margins
        if margin_type in (:rel, :relative)
            abs_margins = (
                _relative_margin(xlims, first(margins)),
                _relative_margin(ylims, last(margins)),
            )
        end
        return (
            _add_margin(xlims, first(abs_margins)), 
            _add_margin(ylims, last(abs_margins)), 
        )
    end
    get_limits(Xs; kwargs...) = _get_limits(Xs; kwargs...)
    get_limits(Xs::Vararg{<:AbstractMatrix}; kwargs...) = _get_limits(Xs; kwargs...)
   
    function box_limits(X)
        (xlims, ylims) = extrema(X; dims=2)
        return (xlims, ylims)
    end

    function envelop((xlims1, ylims1), (xlims2, ylims2))
        xlims = _envelop(xlims1, xlims2)
        ylims = _envelop(ylims1, ylims2)
        return xlims, ylims
    end

    function _envelop(lims1, lims2)
        return (
            min(lims1[1], lims2[1]),
            max(lims1[2], lims2[2]),
        )
    end

    function _relative_margin(lims, percentage)
        w = last(lims) - first(lims)
        return percentage * w
    end
    function _add_margin(lims, margin)
        return (first(lims)-margin, last(lims)+margin)
    end
end