import gspread
from google.auth.transport.requests import AuthorizedSession
from oauth2client.service_account import ServiceAccountCredentials

import pandas as pd


scopes = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive',
]

credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'gspread_cred.json',
    scopes=scopes
)

gsc = gspread.authorize(credentials)


def append_df_to_gs(df, spread_sheet:str):

    sheet = gsc.open(spread_sheet)
    params = {'valueInputOption': 'USER_ENTERED'}
    body = {'values': df.values.tolist()}
    sheet.values_append('!A1:G1', params, body)


def pull_sheet(spread_sheet:str, sheet_name:str):

    sheet = gsc.open(spread_sheet)
    wks = sheet.worksheet(sheet_name)

    return pd.DataFrame(wks.get_all_records())

print(pull_sheet('monster_data', 'Sheet1'))
print(pull_sheet('youtube_data', 'Sheet1'))
