import pandas as pd
import matplotlib.pyplot
import numpy as np
from datetime import datetime
from math import floor


def row_and_field_count(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Returns record count and field count of given data frame; 
    data reference for chaining.
    """
    # Returning
    return (data, { "Record count" : data.shape[0],
                    "Field count"  : data.shape[1] })


def c_records_from_m(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Returns C records, starting from record #M;
    data reference for chaining.
    """
    # Getting arguments
    c = kwargs["c"]
    m = kwargs["m"]

    # Returning
    return (data, { "".join([str(c), " records from ", str(m)]) : data.iloc[m:m + c] })


def each_nth_record(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Splits given data in 2 groups:
        group 1 - all data for 1st half year months;
        group 2 - all data for 2nd half year months;
    and returns:
        N1 = FSTHm * M records from group 1;
        N2 = SNDHm * M records from group 2;
        data reference for chaining.
    """
    # Getting arguments
    m      = kwargs["m"]
    fst_hm = kwargs["fst_hm"]
    snd_hm = kwargs["snd_hm"]

    # Returning
    n1 = m * fst_hm
    n2 = m * snd_hm
    return (data, { "".join(["Each ", str(n1), "th record from all data for 1st half year months"]) : 
                    data[data["CET"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date().month) < 6][::n1],

                    "".join(["Each ", str(n2), "th record from all data for 2nd half year months"]) : 
                    data[data["CET"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date().month) >= 6][::n2] })


def types_of_fields(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Returns type, determined by Pandas, for each column of data;
    data reference for chaining.
    """
    # Returning
    return (data, { "Types of fields" : data.dtypes })


def separate_cet(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Returns SHALLOW copy of data, with single CET column replaced with
    3 new columns: year, month, day.
    """
    # Original CET data
    cet = data["CET"].apply(lambda e: datetime.strptime(e, "%Y-%m-%d").date())

    # Creating new DataFrame
    result = pd.DataFrame.copy(data, deep = False).drop(columns = "CET")

    # Inserting year, month, day separately
    result.insert(0, "Year", cet.map(lambda x: str(x.year)), allow_duplicates = False)
    result.insert(1, "Month", cet.map(lambda x: "{:02d}".format(x.month)), allow_duplicates = False)
    result.insert(2, "Day", cet.map(lambda x: "{:02d}".format(x.day)), allow_duplicates = False)

    # Returning
    return (result, { "Replaced CET" : result })


def no_event_day_count(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Returns count of days with no events;
    data reference for chaining.
    """
    return (data, { "No event day count" : data[data["Events"].isnull()].shape[0]})


def max_mean_wind_speed_day_with_smallest_mean_humidity(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Returns day with smallest mean humidity, its max and mean wind speed;
    data reference for chaining.
    """
    # Find day with smallest mean humidity
    day = data[data["Mean Humidity"] == data["Mean Humidity"].min()]

    # Returning
    return (data, { "Day" : "-".join(day[["Year", "Month", "Day"]].values.tolist()[0]),
                    "Max wind speed" : day["Max Wind SpeedKm/h"].iloc[0],
                    "Mean wind speed" : day["Mean Wind SpeedKm/h"].iloc[0] })


def months_mean_t_f0t5(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Returns all months (YYYY-mm), those mean temperature is [0, 5]
    data reference for chaining.
    """
    # Getting arguments
    min_mean_t = kwargs["min_mean_t"]
    max_mean_t = kwargs["max_mean_t"]

    # Grouping by year and month, counting mean
    ym_group = data[["Year", "Month", "Mean TemperatureC"]].groupby(["Year", "Month"]).mean()

    # Returning
    return (data, { " ".join(["months mean t [", str(min_mean_t), str(max_mean_t), "]"]) :
                    ym_group[ym_group["Mean TemperatureC"].map(lambda x: min_mean_t <= x <= max_mean_t) == True] })


def mean_max_t_for_each_day_all(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Returns mean of "Max TemperatureC" for each day during all years/months (in all);
    data reference for chaining.
    """
    # Returning
    return (data, { "Mean max t for each day during all" : 
                    data[["Day", "Max TemperatureC"]].groupby(["Day"]).mean()})


def count_event_days_in_each_year(data: pd.DataFrame, **kwargs) -> tuple:
    """
    Returns count of days with given event in each year;
    data reference for chaining.
    """
    # Getting arguments
    event = kwargs["event"]

    # Returning
    return (data, { "Foggy days count in each year" : 
                    data[data["Events"] == event][["Year", "Day"]].groupby(["Year"]).count() })


def event_count_column_chart(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Creates column chart of event count.
    Returns axes with plot;
    data reference for chaining.
    """
    # Selecting data to plot
    event_count = data[["Day", "Events"]].groupby("Events") \
                                         .count() \
                                         .rename(columns = { "Day" : "Count" })
    
    # Plotting
    fig, axes = matplotlib.pyplot.subplots(figsize = (12, 7))
    fig.subplots_adjust(left = 0.25)
    axes.barh(event_count.index, event_count["Count"], height = 0.8)
    axes.set_title("Event count", fontsize = 18)
    axes.set_xlabel("Count")
    axes.set_ylabel("Events")
    axes.tick_params(axis = "both", labelsize = 12)
    axes.invert_yaxis()

    # Returning
    return (data, { "event-count" : event_count, "axes" : axes })


def wind_rose(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Creates pie chart, which represents wind rose.
    Returns axes with plot;
    data reference for chaining.
    """
    # Selecting data to plot
    plot_data = pd.DataFrame({"Wind" : data["WindDirDegrees"].apply(lambda x: floor(x / 45) % 8)}).groupby("Wind") \
                                                                                                  .size() \
                                                                                                  .reset_index(name = "Count")
    counts = plot_data["Count"]
    winds = plot_data["Wind"]
    
    # Plotting
    fig, axes = matplotlib.pyplot.subplots(subplot_kw = dict(projection = "polar"))
    axes.set_theta_zero_location("N")
    axes.set_theta_direction(-1)
    axes.grid(True, linestyle = '--', linewidth = 0.7, alpha = 0.5, zorder = 1)
    bars = axes.bar(winds, counts, width = 0.4, align = "edge", zorder = 2)
    axes.set_ylim(0, max(counts))
    axes.set_yticks([])
    axes.set_title("Wind rose")
    
    # Returning
    return (data, { "wind rose" : plot_data, "axes" : axes })


def mean_max_t_min_dewpoint_for_each_month_all(data: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Creates 2 plots: 
        1. mean max t for each month during all years,
        2. mean min dewpoint for each month during all years.
    Returns both plots;
    data reference for chaining.
    """
    # Selecting data to plot
    plot_data = data[["Year", "Month", "Max TemperatureC", "Min DewpointC"]].groupby(["Year", "Month"]) \
                                                                            .mean() \
                                                                            .reset_index()
    # Plotting
    ym = (plot_data["Year"].astype(str) + "-" + plot_data["Month"].astype(str)).values.tolist()
    t_dp = plot_data[["Max TemperatureC", "Min DewpointC"]].to_dict()
    t_dp["Max TemperatureC"] = tuple(dict(t_dp["Max TemperatureC"]).values())
    t_dp["Min DewpointC"] = tuple(dict(t_dp["Min DewpointC"]).values())
    y = np.arange(len(ym))
    width = 0.5
    fig, axes = matplotlib.pyplot.subplots(figsize = (12, 72))
    for k, v in t_dp.items():
        rects = axes.barh(ym, v, width, label = k)
        axes.bar_label(rects, padding = 3)
    axes.set_title('mean max t and mean min dewpoint for each month during all years')
    axes.set_yticks(y + width, ym)
    axes.legend(loc = 'upper left')
    axes.invert_yaxis()

    # Returning
    return (data, { "plot data" : plot_data, "axes" : axes })


def chain(data: pd.DataFrame, *functions) -> None:
    # Chain constants
    kwargs = { "c"          : 5,
               "m"          : 6,
               "fst_hm"     : 500,
               "snd_hm"     : 300,
               "min_mean_t" : 20,
               "max_mean_t" : 22,
               "event"      : "Fog" }
    
    # Running chain
    for f in functions:
        # Calling current function
        out = f(data, **kwargs)

        # Assign transformed data
        data = out[0]

        # Handle other
        print("====| %s |====>\n" % (f.__name__))
        other = out[1]
        for k, v in other.items():
            if isinstance(v, matplotlib.axes._axes.Axes):
                matplotlib.pyplot.savefig(fname = v.get_title())
                matplotlib.pyplot.clf()
                print("|==> plot saved\n")
            else:
                print("|==> ", k, ":\n\n", v, "\n")
        print("====>\n\n\n")


def main():
    """
    Program entrance point.
    """
    # Reading data
    data = pd.read_csv("Weather.csv")

    # Chaining
    chain(data,
          row_and_field_count,
          c_records_from_m,
          each_nth_record,
          types_of_fields,
          separate_cet,
          no_event_day_count,
          max_mean_wind_speed_day_with_smallest_mean_humidity,
          months_mean_t_f0t5,
          mean_max_t_for_each_day_all,
          count_event_days_in_each_year,
          event_count_column_chart,
          wind_rose,
          mean_max_t_min_dewpoint_for_each_month_all)


if __name__ == "__main__":
    main()