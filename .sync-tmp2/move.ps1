$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$out = @()
$log = 'C:\Code\MyKB\.sync-tmp2\move_log.txt'
function Log($m) { $script:out += $m; Add-Content -Path $log -Value $m -Encoding utf8 }
Remove-Item $log -ErrorAction SilentlyContinue
function Get-NS([object]$item) {
  $f = $item.GetFolder()
  if ($f) { return $f }
  return $shell.Namespace($item.Path)
}
function CountRec($ns, $skipNames) {
  $n = 0
  foreach ($i in @($ns.Items())) {
    if ($skipNames -contains $i.Name) { continue }
    if ($i.IsFolder) {
      try { $n += CountRec (Get-NS $i) $skipNames } catch { $n += 0 }
    } else { $n++ }
  }
  return $n
}
try {
  $shell = New-Object -ComObject Shell.Application
  $pc = $shell.Namespace(17)
  $dev = $null
  foreach ($it in @($pc.Items())) {
    if ($it.IsFolder -and $it.Name -notmatch 'WPS' -and $it.Name -notmatch 'Local Disk') { $dev = $it }
  }
  if (-not $dev) { throw 'device not found' }
  $devNS = Get-NS $dev
  $storage = @($devNS.Items())[0]
  $storageNS = Get-NS $storage
  Log ("storage: " + $storage.Name)
  foreach ($c in @($storageNS.Items())) { Log ("  root child: [" + $c.Name + "]") }

  $obs = $storageNS.ParseName('Obsidian')
  if (-not $obs) { throw 'Obsidian folder not found on phone' }
  $obsNS = Get-NS $obs
  $mkb = $obsNS.ParseName('MyKB')
  if (-not $mkb) { throw 'MyKB folder not found under Obsidian' }
  $srcNS = Get-NS $mkb
  $srcCount = CountRec $srcNS @()
  Log ("source Obsidian\MyKB files: " + $srcCount)

  $doc = $storageNS.ParseName('Documents')
  if (-not $doc) { throw 'Documents folder not found on phone' }
  $docNS = Get-NS $doc
  if ($docNS.ParseName('MyKB')) { throw 'Documents\MyKB already exists - remove it first' }
  $docNS.NewFolder('MyKB')
  $dst = $docNS.ParseName('MyKB')
  if (-not $dst) { throw 'cannot create Documents\MyKB' }
  $dstNS = Get-NS $dst
  Log ("destination Documents\MyKB created")

  $items = @($srcNS.Items())
  $expectedTop = $items.Count
  foreach ($it in $items) {
    Log ("copying: " + $it.Name)
    $dstNS.CopyHere($it.Path, 84)
  }

  $stable = 0
  $deadline = (Get-Date).AddMinutes(25)
  while ($true) {
    Start-Sleep -Seconds 5
    $c = @($dstNS.Items()).Count
    Log ("poll: " + $c + "/" + $expectedTop)
    if ($c -ge $expectedTop) { $stable++; if ($stable -ge 3) { break } } else { $stable = 0 }
    if ((Get-Date) -gt $deadline) { throw 'timeout waiting for copy' }
  }
  $dstCount = CountRec $dstNS @()
  Log ("verify: src=" + $srcCount + " dst=" + $dstCount)
  if ($dstCount -ne $srcCount) { throw ('count mismatch: ' + $srcCount + ' vs ' + $dstCount) }
  Log ("MOVE_COPY_OK")
} catch {
  Log ("ERROR: " + $_.Exception.ToString())
}
$out | Out-File -FilePath 'C:\Code\MyKB\.sync-tmp2\move_result.txt' -Encoding utf8
