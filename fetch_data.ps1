$ErrorActionPreference = 'Stop'

$dataDir = Join-Path $PSScriptRoot 'data'
$splitDir = Join-Path $dataDir 'splits'
New-Item -ItemType Directory -Path $splitDir -Force | Out-Null

function Get-VerifiedFile {
    param([string]$Uri, [string]$Destination, [string]$Sha256)

    if ((Test-Path -LiteralPath $Destination -PathType Leaf) -and
        ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -eq $Sha256)) {
        Write-Host "Verified $Destination"
        return
    }

    $partial = "$Destination.part"
    & curl.exe --fail --location --retry 2 --output $partial $Uri
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $Uri"
    }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $Sha256) {
        throw "SHA-256 mismatch for $Uri"
    }
    Move-Item -LiteralPath $partial -Destination $Destination -Force
}

$archive = 'https://huggingface.co/datasets/marsyas/gtzan/resolve/7031b10/data/genres.tar.gz?download=true'
Get-VerifiedFile $archive (Join-Path $dataDir 'genres.tar.gz') 'B28CC067FF6199BD826F5D1A6931458586D64ACAE7F44B2E882B7A97AF057531'

$splitRoot = 'https://raw.githubusercontent.com/boblsturm/GTZAN/main'
$hashes = @{
    train = '6252B4928F0A29185B894AED5BABADAE41104B4865F00AEFF9B79DE1C491F45C'
    valid = '2E0A0E0B513AB61AB206EE4E6FB5F18A41D26E469B160A765196CB47841E3B7E'
    test = 'CE78134675F23D02CA4B7F73876C5778B36D8C2FBCCCBE54A0359D1A2D86BDEF'
}
foreach ($split in @('train', 'valid', 'test')) {
    $filename = "${split}_filtered.txt"
    Get-VerifiedFile "$splitRoot/$filename" (Join-Path $splitDir $filename) $hashes[$split]
}
