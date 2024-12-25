# orchestration/cli.py
import click
from pathlib import Path
import json

@click.group()
def cli():
    """Knowledge Graph Creation CLI"""
    pass

@cli.command()
@click.argument('config_path', type=click.Path(exists=True))
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--pattern', '-p', default="*.pdf,*.docx,*.csv", 
              help="File patterns to process (comma-separated)")
@click.option('--output', '-o', type=click.Path(), 
              help="Output path for processing summary")
def process(config_path: str, input_dir: str, pattern: str, output: Optional[str]):
    """Process documents and create knowledge graph."""
    try:
        manager = PipelineManager(config_path)
        results = manager.process_directory(input_dir, pattern)
        
        # Print summary to console
        click.echo(f"Processing completed:")
        click.echo(f"Total documents: {results['total_documents']}")
        click.echo(f"Successful: {results['successful']}")
        click.echo(f"Failed: {results['failed']}")
        click.echo(f"Total entities: {results['total_entities']}")
        click.echo(f"Total relationships: {results['total_relationships']}")
        
        # Save detailed results if output path provided
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            click.echo(f"Detailed results saved to: {output_path}")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == "__main__":
    cli()