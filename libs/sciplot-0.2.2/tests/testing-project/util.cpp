#include <sciplot/sciplot.hpp>
using namespace sciplot;

void plot()
{
    const Vec x = linspace(0.0, 5.0, 8);
    const Vec y = linspace(2.0, 6.0, 8);

    Plot plot;

    plot.palette("dark2");

    plot.samples(200);

    plot.drawCurve(x, y).label("A");

    plot.show();
}
