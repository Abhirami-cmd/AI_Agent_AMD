$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$dataDir = Join-Path $repoRoot "data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

function XmlEscape([string]$value) {
    return [System.Security.SecurityElement]::Escape($value)
}

function ColumnName([int]$index) {
    $name = ""
    while ($index -gt 0) {
        $mod = ($index - 1) % 26
        $name = [char](65 + $mod) + $name
        $index = [math]::Floor(($index - $mod) / 26)
    }
    return $name
}

function SheetXml($headers, $rows) {
    $xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    $xml += '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
    $rowNumber = 1
    $xml += "<row r=`"$rowNumber`">"
    for ($i = 0; $i -lt $headers.Count; $i++) {
        $cell = "$(ColumnName ($i + 1))$rowNumber"
        $xml += "<c r=`"$cell`" t=`"inlineStr`"><is><t>$(XmlEscape $headers[$i])</t></is></c>"
    }
    $xml += "</row>"

    foreach ($row in $rows) {
        $rowNumber += 1
        $xml += "<row r=`"$rowNumber`">"
        for ($i = 0; $i -lt $headers.Count; $i++) {
            $cell = "$(ColumnName ($i + 1))$rowNumber"
            $value = [string]$row[$i]
            $xml += "<c r=`"$cell`" t=`"inlineStr`"><is><t>$(XmlEscape $value)</t></is></c>"
        }
        $xml += "</row>"
    }
    $xml += '</sheetData></worksheet>'
    return $xml
}

function WriteUtf8([string]$path, [string]$value) {
    [IO.File]::WriteAllText($path, $value, [Text.Encoding]::UTF8)
}

$incidentsHeaders = @("incident_id", "title", "service", "severity", "started_at", "description")
$incidentsRows = @(
    @("INC-001", "Checkout latency and payment failures", "checkout-service", "Critical", "2026-06-10 10:15", "Checkout requests are timing out and payment authorization failures started after database write latency increased."),
    @("INC-002", "Inventory API elevated 5xx errors", "inventory-api", "High", "2026-06-10 11:35", "Inventory reads are failing after a deployment introduced a higher error rate while infrastructure signals stayed mostly stable.")
)

$dependenciesHeaders = @("incident_id", "source", "dependency", "tower")
$dependenciesRows = @(
    @("INC-001", "checkout-service", "payment-api", "application"),
    @("INC-001", "checkout-service", "payment-db", "storage"),
    @("INC-001", "payment-api", "node-pool-a", "compute"),
    @("INC-001", "payment-api", "east-lb", "network"),
    @("INC-002", "inventory-api", "catalog-db", "storage"),
    @("INC-002", "inventory-api", "node-pool-b", "compute"),
    @("INC-002", "inventory-api", "west-lb", "network")
)

$telemetryHeaders = @("incident_id", "timestamp", "tower", "signal", "value", "baseline", "unit", "component")
$telemetryRows = @(
    @("INC-001", "2026-06-10 09:55", "application", "checkout_p95_latency_ms", "180", "220", "ms", "checkout-service"),
    @("INC-001", "2026-06-10 10:00", "application", "checkout_p95_latency_ms", "195", "220", "ms", "checkout-service"),
    @("INC-001", "2026-06-10 10:05", "application", "checkout_p95_latency_ms", "210", "220", "ms", "checkout-service"),
    @("INC-001", "2026-06-10 10:10", "application", "checkout_p95_latency_ms", "420", "220", "ms", "checkout-service"),
    @("INC-001", "2026-06-10 10:15", "application", "checkout_p95_latency_ms", "950", "220", "ms", "checkout-service"),
    @("INC-001", "2026-06-10 10:20", "application", "checkout_p95_latency_ms", "1200", "220", "ms", "checkout-service"),
    @("INC-001", "2026-06-10 10:25", "application", "checkout_p95_latency_ms", "980", "220", "ms", "checkout-service"),
    @("INC-001", "2026-06-10 09:55", "application", "checkout_error_rate_pct", "0.4", "1.0", "%", "checkout-service"),
    @("INC-001", "2026-06-10 10:00", "application", "checkout_error_rate_pct", "0.5", "1.0", "%", "checkout-service"),
    @("INC-001", "2026-06-10 10:05", "application", "checkout_error_rate_pct", "0.7", "1.0", "%", "checkout-service"),
    @("INC-001", "2026-06-10 10:10", "application", "checkout_error_rate_pct", "2.2", "1.0", "%", "checkout-service"),
    @("INC-001", "2026-06-10 10:15", "application", "checkout_error_rate_pct", "8.9", "1.0", "%", "checkout-service"),
    @("INC-001", "2026-06-10 10:20", "application", "checkout_error_rate_pct", "12.5", "1.0", "%", "checkout-service"),
    @("INC-001", "2026-06-10 10:25", "application", "checkout_error_rate_pct", "9.6", "1.0", "%", "checkout-service"),
    @("INC-001", "2026-06-10 09:55", "storage", "payment_db_write_latency_ms", "8", "15", "ms", "payment-db"),
    @("INC-001", "2026-06-10 10:00", "storage", "payment_db_write_latency_ms", "10", "15", "ms", "payment-db"),
    @("INC-001", "2026-06-10 10:05", "storage", "payment_db_write_latency_ms", "12", "15", "ms", "payment-db"),
    @("INC-001", "2026-06-10 10:10", "storage", "payment_db_write_latency_ms", "130", "15", "ms", "payment-db"),
    @("INC-001", "2026-06-10 10:15", "storage", "payment_db_write_latency_ms", "260", "15", "ms", "payment-db"),
    @("INC-001", "2026-06-10 10:20", "storage", "payment_db_write_latency_ms", "310", "15", "ms", "payment-db"),
    @("INC-001", "2026-06-10 10:25", "storage", "payment_db_write_latency_ms", "190", "15", "ms", "payment-db"),
    @("INC-001", "2026-06-10 09:55", "compute", "node_pool_cpu_pct", "52", "65", "%", "node-pool-a"),
    @("INC-001", "2026-06-10 10:15", "compute", "node_pool_cpu_pct", "74", "65", "%", "node-pool-a"),
    @("INC-001", "2026-06-10 10:20", "network", "east_lb_packet_loss_pct", "0.6", "0.3", "%", "east-lb"),
    @("INC-002", "2026-06-10 11:15", "application", "inventory_5xx_rate_pct", "0.3", "1.0", "%", "inventory-api"),
    @("INC-002", "2026-06-10 11:30", "application", "inventory_5xx_rate_pct", "7.8", "1.0", "%", "inventory-api"),
    @("INC-002", "2026-06-10 11:35", "application", "inventory_5xx_rate_pct", "10.4", "1.0", "%", "inventory-api"),
    @("INC-002", "2026-06-10 11:35", "application", "deployment_errors", "15", "1", "count", "inventory-api"),
    @("INC-002", "2026-06-10 11:35", "storage", "catalog_db_read_latency_ms", "24", "18", "ms", "catalog-db"),
    @("INC-002", "2026-06-10 11:35", "compute", "node_pool_memory_pct", "70", "75", "%", "node-pool-b"),
    @("INC-002", "2026-06-10 11:35", "network", "west_lb_latency_ms", "28", "25", "ms", "west-lb")
)

$tmp = Join-Path $env:TEMP ("amd_rca_xlsx_" + [guid]::NewGuid().ToString())
New-Item -ItemType Directory -Force -Path $tmp | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $tmp "_rels") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $tmp "xl") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $tmp "xl\_rels") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $tmp "xl\worksheets") | Out-Null

WriteUtf8 (Join-Path $tmp "[Content_Types].xml") '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet3.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>'
WriteUtf8 (Join-Path $tmp "_rels\.rels") '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'
WriteUtf8 (Join-Path $tmp "xl\workbook.xml") '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="incidents" sheetId="1" r:id="rId1"/><sheet name="dependencies" sheetId="2" r:id="rId2"/><sheet name="telemetry" sheetId="3" r:id="rId3"/></sheets></workbook>'
WriteUtf8 (Join-Path $tmp "xl\_rels\workbook.xml.rels") '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/><Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet3.xml"/></Relationships>'
WriteUtf8 (Join-Path $tmp "xl\worksheets\sheet1.xml") (SheetXml $incidentsHeaders $incidentsRows)
WriteUtf8 (Join-Path $tmp "xl\worksheets\sheet2.xml") (SheetXml $dependenciesHeaders $dependenciesRows)
WriteUtf8 (Join-Path $tmp "xl\worksheets\sheet3.xml") (SheetXml $telemetryHeaders $telemetryRows)

$xlsxPath = Join-Path $dataDir "observability_sample.xlsx"
$zipPath = Join-Path $dataDir "observability_sample.zip"
if (Test-Path $xlsxPath) { Remove-Item $xlsxPath -Force }
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path (Join-Path $tmp "*") -DestinationPath $zipPath -Force
Move-Item -Path $zipPath -Destination $xlsxPath -Force
Remove-Item $tmp -Recurse -Force

$pdfText = "Cross-Tower RCA Runbook`n`nStart from the incident workflow. Use telemetry from compute, storage, network, and application towers. Storage latency that precedes application timeout errors is a strong RCA candidate. Deployment error spikes can indicate an application regression. Network packet loss must be considered when dependency calls fail across services. Compute pressure is relevant when CPU, memory, restart, or node health anomalies align with the incident window. Always show confidence, evidence, alternatives, and validation steps."
$escapedPdfText = $pdfText.Replace("\", "\\").Replace("(", "\(").Replace(")", "\)").Replace("`n", "\n")
$objects = @(
    "1 0 obj`n<< /Type /Catalog /Pages 2 0 R >>`nendobj`n",
    "2 0 obj`n<< /Type /Pages /Kids [3 0 R] /Count 1 >>`nendobj`n",
    "3 0 obj`n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>`nendobj`n",
    "4 0 obj`n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>`nendobj`n"
)
$stream = "BT /F1 11 Tf 54 730 Td ($escapedPdfText) Tj ET"
$objects += "5 0 obj`n<< /Length $($stream.Length) >>`nstream`n$stream`nendstream`nendobj`n"
$pdf = "%PDF-1.4`n"
$offsets = @(0)
foreach ($object in $objects) {
    $offsets += [Text.Encoding]::ASCII.GetByteCount($pdf)
    $pdf += $object
}
$xrefOffset = [Text.Encoding]::ASCII.GetByteCount($pdf)
$pdf += "xref`n0 $($objects.Count + 1)`n0000000000 65535 f `n"
foreach ($offset in $offsets[1..($offsets.Count - 1)]) {
    $pdf += ("{0:D10} 00000 n `n" -f $offset)
}
$pdf += "trailer`n<< /Size $($objects.Count + 1) /Root 1 0 R >>`nstartxref`n$xrefOffset`n%%EOF"
[IO.File]::WriteAllText((Join-Path $dataDir "reference_runbook.pdf"), $pdf, [Text.Encoding]::ASCII)

Write-Host "Created $xlsxPath"
Write-Host "Created $(Join-Path $dataDir 'reference_runbook.pdf')"
