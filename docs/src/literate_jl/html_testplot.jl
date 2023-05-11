using Pkg
Pkg.activate(joinpath(@__DIR__,"..", ".."))
using WGLMakie, JSServe
WGLMakie.activate!()
#%%
function make_fig()
    dat = rand(2, 10)
    fig = Figure(; resolution=(500, 350))
    ax = Axis(fig[1,1])
    toggles = [
        Toggle(fig, active=true),
    ]
    #=labels = [
        Label(fig, "s1 visible"),
        Label(fig, "s1 invisible")
    fig[1, 2] = grid!(hcat(toggles, labels), tellheight = false)
    ]=#

    fig[1, 2] = grid!(hcat(toggles), tellheight = false)
    s1 = scatter!(ax, dat; label="s1", visible=true)
    connect!(s1.visible, toggles[1].active)
    axislegend(ax)
    
    #scatter!(ax, dat)
    return fig
end
#make_fig()
#%%
open(joinpath(ENV["HOME"], "Pictures", "testplot.html"), "w") do io
    println(io, """
    <html>
        <head>
        </head>
        <body>
    """)
    Page(exportable=true, offline=true)
    # Then, you can just inline plots or whatever you want :)
    show(io, MIME"text/html"(), make_fig())
    println(io, """
        </body>
    </html>
    """)
end