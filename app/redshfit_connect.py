import redshift_connector
from config import Config
import pandas as pd


def get_all_data_rs(table_name : str):

    conn = redshift_connector.connect(
        host=Config.host,
        database=Config.dbname,
        user=Config.user,
        password=Config.password
    )

    cursor: redshift_connector.Cursor = conn.cursor()

    cursor.execute(f"select * from btc.{table_name};")
    data : pd.DataFrame = cursor.fetch_dataframe()
    conn.close()

    return data