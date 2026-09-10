import pandas as pd
import plotly.express as px


def create_chart(df, chart_type, x_column, y_column, title):

    # Aggregate data for categorical charts
    if chart_type == "pie":
        chart_df = (
            df.groupby(x_column)[y_column]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            chart_df,
            names=x_column,
            values=y_column,
            title=title
        )

    elif chart_type == "bar":
        chart_df = (
            df.groupby(x_column)[y_column]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            chart_df,
            x=x_column,
            y=y_column,
            title=title,
            text=y_column
        )

    elif chart_type == "line":
        fig = px.line(
            df,
            x=x_column,
            y=y_column,
            title=title
        )

    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    return fig