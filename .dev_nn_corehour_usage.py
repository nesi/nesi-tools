#!/usr/bin/env python3

import pandas as pd
from datetime import datetime, timedelta
import argparse
import os
import sys
from dateutil import parser
import subprocess

# COLUMN_ALIAS = {
#     "time": "period",
#     "cpu": "CPU",
#     "core": "CPU",
#     "GPU:A100": "A100",
#     "gres/gpu:a100": "A100",
#     "GPU:A100-1G.5GB": "A100",
#     "GPU:P100": "P100",
#     "gres/gpu:p100": "P100",
#     "node": "Node",
#     "mem": "Memory",
#     "memGB": "Memory",
# }

HEADER_FORMAT = {
    "hours": {
        "cpu": "CPU/Hours",
        "a100": "A100/Hours",
        "p100": "P100/Hours",
        "mem": "Memory/Byte Hours",
        "node": "Node/Hours",
        "total": "Total/Units",
    },
    "dollars": {
        "cpu": "CPU/NZD",
        "a100": "A100/NZD",
        "p100": "P100/NZD",
        "mem": "Memory/NZD",
        "node": "Node/NZD",
        "total": "Total/NZD",
    },
    "units": {
        "cpu": "CPU/Units",
        "a100": "A100/Units",
        "p100": "P100/Units",
        "mem": "Memory/Units",
        "node": "Node/Units",
        "total": "Total/Units",
    },
}

# List of columns to show on each cluster.
COLUMN_LIST = {
    "maui": ["node"],
    "maui_ancil": ["cpu", "p100", "mem"],
    "mahuika": ["cpu", "a100", "p100", "mem"],
}

# Below could be part of formatter.
COLUMN_SPACE = {
    "period": 40,
    "cpu": 14,
    "a100": 14,
    "p100": 14,
    "mem": 14,
    "node": 14,
    "total": 14,
}

# Factor to apply to output numbers.
unit_to_dollar = 0.02

CONVERSION_RATES = {
    "units": {
        "cpu": 0.7,
        "p100": 7,
        "a100": 18,
        "mem": 0.0000000001,
        "node": 40,
    },
    "hours": {"mem": 1000000},
}

CONVERSION_RATES["dollars"] = {
    k: v * unit_to_dollar for k, v in CONVERSION_RATES["units"].items()
}


def convert_si(n):
    """
    Output formatter.
    """
    for prefix in ["", "k", "M", "G", "T"]:
        if n > 99999:
            n /= 1000
        else:
            break

    return "{0:.6g}{1}".format(n, prefix)


def main():
    args = parse_input()

    print(
        f"Usage for {','.join(args.account)}{' by '+','.join(args.user) if any(args.user) else ''}"
    )

    row_formatters = {
        "period": lambda x: "{0} {1}".format(
            x.strftime("%Y-%m-%d"),
            (x + timedelta(days=args.time_interval)).strftime("%Y-%m-%d"),
        ),
        "cpu": convert_si,
        "a100": convert_si,
        "p100": convert_si,
        "mem": convert_si,
        "node": convert_si,
        "total": convert_si,
    }

    # Get list of clusters.

    # Stupid waste of time method.
    # result = subprocess.run(
    #         ["sacctmgr", "--json", "-n", "-p", "list", "clusters"],
    #         stdout=subprocess.PIPE,
    #         timeout=30,
    #         check=True,
    #     )
    # assert not result.returncode

    # clusters = json.loads(result.stdout)["clusters"]

    clusters = ["mahuika", "maui", "maui_ancil"]

    # Decide method to get data.
    if args.datalake:
        try:
            source_method = get_from_datalake
        except:
            print("Could not access datalake, falling back to sreport method.")
            source_method = get_from_sreport
    else:
        source_method = get_from_sreport

    # Times in datetime format.
    start_datetime = (
        parser.parse(args.start_date)
        if args.start_date
        else datetime.now() - timedelta(days=360)
    )
    end_datetime = parser.parse(args.end_date) if args.end_date else datetime.now()

    for cluster in clusters:
        print(f"Usage for {cluster} in {args.unit.capitalize()}")

        row_date = start_datetime
        next_row_date = row_date + timedelta(days=args.time_interval)

        df_columns = ["period", *COLUMN_LIST[cluster]]

        # No point in summing non equivalent units
        if args.unit in ["units", "dollars"]:
            df_columns.append("total")

        # One dataframe per cluster.
        df_main = pd.DataFrame(columns=df_columns)

        print_header = True
        while next_row_date < end_datetime:
            # Output from 'source method' should be rows of tresable objects,
            # with the standard header names.
            # print(source_method(args, cluster, row_date, next_row_date))

            # If database method fails, go back to sreport.
            while True:
                try:
                    df = source_method(args, cluster, row_date, next_row_date)
                    break
                except Exception as e:
                    if source_method.__name__ == "get_from_database":
                        print(f"Database call failed, :{str(e)}")
                        source_method = get_from_sreport
                        print("Falling back to sreport.")
                    else:
                        raise e

            df = df.groupby(["period", "facility", "metric"], as_index=False).sum()
            df = df.pivot(
                index="period", columns="metric", values="value"
            ).reset_index()

            # Drop unused column. (should only be needed if unexpected addition of gres)
            df = df[df.columns.intersection(df_columns)]
            # Reset index. Makes column formatting easier.

            # Drop unused columns
            # df = df[df_main.columns.intersection(df.columns)]

            # Only makes sense so sum if converted.
            if args.unit in ["units", "dollars"]:
                df["total"] = df.sum(numeric_only=True, axis=1)

            # Apply conversion
            for column, factor in CONVERSION_RATES[args.unit].items():
                if column in df.columns:
                    df[column] = df[column].apply(lambda x: x * factor)

            df_main = pd.concat([df_main, df])  # .reset_index()
            output_human(df_main, args, print_header, row_formatters)

            print_header = False
            row_date = next_row_date
            next_row_date = row_date + timedelta(days=args.time_interval)

            # tmin = df.iloc[0]["period"]
            # tmax = df.iloc[-1]["period"] + timedelta(days=args.time_interval)

        # Add total
        # df_main = pd.concat([df_main, df_main[df_main.columns[1:]].sum()])
        # # I hate pandas
        # print(
        #     df_main.tail(1).to_string(
        #         index=False,
        #         header=False,
        #         col_space={x: COLUMN_SPACE[x] for x in df_main.columns},
        #         formatters=row_formatters,
        #     )
        # )


def parse_input():
    progname = os.path.basename(__file__)

    argparser = argparse.ArgumentParser(
        prog=progname, description="Prints a summary of resource usage."
    )
    # parser.add_argument("-c","--calendar-months", help="Does nothing.", action="store_true", default=False)
    # parser.add_argument("-n", "--number-months <n_months>", help="How many months history to check. (same as \`-S now -<n_months> months\`)", action='store_true')
    argparser.add_argument(
        "-d",
        "--datalake",
        help="Use numbers from datalake if possible (faster).",
        action="store_true",
        default=False,
    )
    # Below is version of -u that takes multiple arguments.
    # Removed for now as casused issues.
    # argparser.add_argument("-u", "--user", help="Show only compute units used by user(s).", default="", nargs="+")
    argparser.add_argument(
        "-u", "--user", help="Show only compute units used by user(s).", default=""
    )
    # argparser.add_argument(
    #     "-t", "--time-format", help="Time format string.", default="%Y-%m-%d"
    # )
    argparser.add_argument(
        "-n",
        "--unit",
        choices=CONVERSION_RATES.keys(),
        help="Show usage in dollar values (approx).",
        default="units",
    )
    argparser.add_argument(
        "-S",
        "--start-date",
        help="Select records from after the specified time. (default = -1 year)",
        default=None,
    )
    argparser.add_argument(
        "-E",
        "--end-date",
        help="Select records from before the specified time. (default = now)",
        default=None,
    )
    argparser.add_argument(
        "-i",
        "--time-interval",
        help="Size of time bucket, in days.",
        type=int,
        default="30",
    )
    # Required if -u not given
    argparser.add_argument(
        "account",
        metavar="account",
        nargs="*",
    )

    args = argparser.parse_args()

    # If no accounts specified, assume -u set to user.
    if len(args.account) < 1 and args.user == "":
        args.user = os.getlogin()

    # Remove this if multiarg -u is fixed.
    args.user = [args.user]

    return args


def get_from_datalake(args, cluster, start_datetime, end_datetime):
    from NeSI.data import datalake

    user_filter = (
        "AND u.login IN ( '" + "' , '".join(args.user) + "' )" if any(args.user) else ""
    )

    account_filter = (
        "AND u.account IN ( '" + "' , '".join(args.account) + "' )"
        if any(args.account)
        else ""
    )

    SQL = f"""
    SELECT
    facility,
    r.tres_name AS metric,
    coalesce(sum(service) / count(DISTINCT start), 0) AS value,
    u.login AS user,
    u.account AS account
    FROM usage.hourly AS u
    INNER JOIN usage.resource_type AS r ON r.tres_name = u.tres
    WHERE
    tres != 'billing'
    {user_filter}
    {account_filter}
    AND u.facility = '{cluster}'
    AND start BETWEEN '{start_datetime.isoformat()}'
    AND '{end_datetime.isoformat()}'
    GROUP BY facility, metric, u.login, u.account
    """
    result = datalake.query(SQL)
    result["period"] = start_datetime
    # # Drop unused columns.
    # if cluster in COLUMN_LIST:
    #     df.drop(labels=column_mask[cluster], axis=1, inplace=True, errors="ignore")

    resource_map = {
        "core": "cpu",
        "GPU:A100": "a100",
        "GPU:P100": "p100",
        "GPU:A100-1G.5GB": "a100",
        "memGB": "mem",
    }

    # Apply map
    result["metric"].replace(resource_map, inplace=True)
    if result.empty:
        result.loc[0] = [cluster, "cpu", 0, "", "", start_datetime]
    return result


def get_from_sreport(args, cluster, start_datetime, end_datetime):
    sreport_cmd = [
        "sreport",
        "-Pn",
        "-M",
        cluster,
        "cluster",
        "AccountUtilizationByUser",
        "-t",
        "Hours",
        "-T",
        "time,cpu,gres/gpu:a100,gres/gpu:p100,mem,node",
        f"Start={start_datetime.strftime('%Y-%m-%d')}",
        f"End={end_datetime.strftime('%Y-%m-%d')}",
    ]

    if any(args.user):
        sreport_cmd += [f"User={','.join(args.user)}"]

    if args.account:
        sreport_cmd += [f"Account={','.join(args.account)}"]

    # print(" ".join(sreport_cmd))

    sreport_out = subprocess.run(
        sreport_cmd,
        stdout=subprocess.PIPE,
        timeout=30,
        text=True,
        check=True,
    )
    if sreport_out.returncode:
        raise Exception(sreport_out.stderr)

    df = pd.DataFrame(
        columns=["facility", "account", "user", "email", "metric", "value", "period"]
    )
    pd_iloc = 0

    # Maybe vectorzed way to do this faster?
    for row in sreport_out.stdout.strip().split("\n"):
        # Sreport will say nothing if no usage over period.
        if row:
            df.loc[pd_iloc] = [*row.split("|"), start_datetime]
        else:
            df.loc[pd_iloc] = [
                cluster,
                "",
                args.user[0],
                "",
                "cpu",
                "0",
                start_datetime,
            ]

        pd_iloc += 1

    resource_map = {
        "core": "cpu",
        "gres/gpu:a100": "a100",
        "gres/gpu:p100": "p100",
    }

    df["metric"].replace(resource_map, inplace=True)

    # df.loc[df["user"].isin(args.user)]

    # Filter by user and return.

    # Convert numeric strings
    df["value"] = df["value"].apply(pd.to_numeric, errors="ignore")
    # df = df.groupby()
    return df

    # This is the original function to fetch all data in one go.
    # Current version uses multiple calls for simplicity of code.
    # This version is faster.

    # def get_from_datalake():
    #     SQL = f"""
    #     SELECT
    #     time_bucket_gapfill('{args.time_interval}d', start) AS time,
    #     facility,
    #     r.tres_name as metric,
    #     coalesce(sum(service) / count(DISTINCT start), 0) AS value
    #     FROM usage.hourly AS u
    #     INNER JOIN usage.resource_type AS r ON r.tres_name = u.tres
    #     WHERE
    #     tres != 'billing'
    #     {"u.user in ('" + ', '''.join(args.account) + "')" if args.user else "" }
    #     AND u.account IN ( '{ ', '''.join(args.account) }' )
    #     AND start BETWEEN '{start_datetime.isoformat()}' AND '{end_datetime.isoformat()}'
    #     GROUP BY metric, facility, time
    #     """
    #     return  NeSI.data.datalake.query(SQL)

    #  for group_name, df_group in j.groupby("facility"):

    # Old 'time formatter' function
    # def time_formatter(t, interval=args.time_interval):
    #     # Format datetime to input spec.

    #     if args.time_interval == 1:
    #         return t.strftime(args.time_format)
    #     return (
    #         t.strftime(args.time_format)
    #         + " -> "
    #         + (t + timedelta(days=args.time_interval)).strftime(args.time_format)
    #     )
    # .tail(1)


def output_human(df_main, args, print_header, row_formatters):
    print(
        df_main.tail(1)
        .rename(columns=HEADER_FORMAT[args.unit])
        .fillna(0)
        .to_string(
            index=False,
            header=print_header,
            col_space=list(COLUMN_SPACE[x] for x in df_main.columns),
            justify="initial",
            formatters=list(row_formatters[x] for x in df_main.columns),
        )
    )


def output_json(df_main):
    raise Exception("Feature not implimented.")


def output_csv(df_main):
    raise Exception("Feature not implimented.")


if __name__ == "__main__":
    main()
