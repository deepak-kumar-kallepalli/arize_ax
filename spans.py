from arize import ArizeClient
from config import AI_INTEGRATION_ID, JUDGE_MODEL_NAME, SPACE_ID


def list_spans(client: ArizeClient, project_name, start_time, end_time):
    df = client.spans.export_to_df(
        project_name=project_name,
        space_id=SPACE_ID,
        start_time=start_time,
        end_time=end_time
    )

    print(f"total spans in the project: {len(df)}")
    print(f"df columns: {df.columns}")
    print(df.loc[0])