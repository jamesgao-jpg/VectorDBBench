from html import escape

from vectordb_bench.frontend.config.styles import DB_SELECTOR_COLUMNS, DB_TO_ICON
from vectordb_bench.frontend.config.dbCaseConfigs import DB_LIST
import streamlit as st


def dbSelector(st: st):
    st.markdown(
        "<div style='height: 12px;'></div>",
        unsafe_allow_html=True,
    )
    st.subheader("STEP 1: Select the database(s)")
    st.markdown(
        "<div style='color: #647489; margin-bottom: 24px; margin-top: -12px;'>Choose at least one database to test.</div>",
        unsafe_allow_html=True,
    )

    dbIsActived = {}

    for row_start in range(0, len(DB_LIST), DB_SELECTOR_COLUMNS):
        row_columns = st.columns(DB_SELECTOR_COLUMNS, gap="small")
        row_dbs = DB_LIST[row_start : row_start + DB_SELECTOR_COLUMNS]

        for column, db in zip(row_columns, row_dbs):
            image_src = DB_TO_ICON[db]
            column.markdown(
                (
                    '<div style="height:112px;display:flex;align-items:center;justify-content:center;'
                    'margin-bottom:8px;">'
                    f'<img src="{escape(image_src, quote=True)}" '
                    f'alt="{escape(db.name, quote=True)} logo" '
                    'style="width:100px;height:100px;object-fit:contain;object-position:center;">'
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            dbIsActived[db] = column.checkbox(db.name)
    activedDbList = [db for db in DB_LIST if dbIsActived[db]]

    return activedDbList
