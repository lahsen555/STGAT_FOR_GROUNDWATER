"""Relocation must preserve provenance checks and reject modified source inputs."""
from pathlib import Path
import hashlib,json,os,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stgnn_data import resolve_imputation_cutoff

class ReleasePathTests(unittest.TestCase):
    def test_relative_merge_audit_works_outside_the_repository_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            data=Path(td)/'relocated/data';data.mkdir(parents=True)
            source=data/'source.csv';source.write_bytes(b'well_id,date\na,2000-01\n')
            merged=data/'merged.csv';merged.write_bytes(b'placeholder')
            audit={'input_sha256':{'source.csv':hashlib.sha256(source.read_bytes()).hexdigest()}}
            (data/'merged_audit.json').write_text(json.dumps(audit))
            cutoff,details=resolve_imputation_cutoff(merged,None,None)
            self.assertIsNone(cutoff)
            self.assertTrue(details['merge_sources_verified'])
            source.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'Merge sources changed'):
                resolve_imputation_cutoff(merged,None,None)

if __name__=='__main__':unittest.main()
