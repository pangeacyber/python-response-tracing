from __future__ import annotations

import click
from dotenv import load_dotenv
from openai import OpenAI
from pangea import PangeaConfig
from pangea.services import Audit

load_dotenv(override=True)


@click.command()
@click.option("--model", default="gpt-4o-mini", show_default=True, required=True, help="OpenAI model.")
@click.option(
    "--audit-token",
    envvar="PANGEA_AUDIT_TOKEN",
    required=True,
    help="Pangea Secure Audit Log API token. May also be set via the `PANGEA_AUDIT_TOKEN` environment variable.",
)
@click.option(
    "--audit-config-id",
    envvar="PANGEA_AUDIT_CONFIG_ID",
    required=False,
    help="Pangea Secure Audit Log configuration ID. May also be set via the `PANGEA_AUDIT_CONFIG_ID` environment variable.",
)
@click.option(
    "--pangea-domain",
    envvar="PANGEA_DOMAIN",
    default="aws.us.pangea.cloud",
    show_default=True,
    required=True,
    help="Pangea API domain. May also be set via the `PANGEA_DOMAIN` environment variable.",
)
@click.option(
    "--openai-api-key",
    envvar="OPENAI_API_KEY",
    required=True,
    help="OpenAI API key. May also be set via the `OPENAI_API_KEY` environment variable.",
)
@click.argument("prompt")
def main(
    *,
    prompt: str,
    model: str,
    audit_token: str,
    audit_config_id: str | None = None,
    pangea_domain: str,
    openai_api_key: str,
) -> None:
    config = PangeaConfig(domain=pangea_domain)
    audit = Audit(token=audit_token, config=config, config_id=audit_config_id)

    # Generate chat completions.
    completion = OpenAI(api_key=openai_api_key).chat.completions.create(
        messages=[{"role": "user", "content": prompt}], model=model, stream=False
    )
    contents = (x.message.content for x in completion.choices if x.message.content)

    # Log response.
    audit.log_bulk(
        [{"event_type": "inference:response", "event_tools": OpenAI.__name__, "event_output": x} for x in contents]
    )

    # Print response.
    for content in contents:
        click.echo(content)


if __name__ == "__main__":
    main()
