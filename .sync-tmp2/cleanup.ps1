$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$out = @()
function Log($m) { $script:out += $m }
function Get-NS([object]$item) {
  $f = $item.GetFolder()
  if ($f) { return $f }
  return $shell.Namespace($item.Path)
}
function CountRec($ns) {
  $n = 0
  foreach ($i in @($ns.Items())) {
    if ($i.IsFolder) {
      try { $n += CountRec (Get-NS $i) } catch { $n += 0 }
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
  $devNS = Get-NS $dev
  $storageNS = Get-NS (@($devNS.Items())[0])

  # verify good copy
  $obsNS = Get-NS ($storageNS.ParseName('Obsidian'))
  $mkb = $obsNS.ParseName('MyKB')
  if ($mkb) {
    $c = CountRec (Get-NS $mkb)
    Log ("Obsidian\MyKB files = " + $c + " (expect 169)")
  } else { Log "Obsidian\MyKB MISSING!" }

  # inspect & remove leftover
  $docNS = Get-NS ($storageNS.ParseName('Documents'))
  $left = $docNS.ParseName('MyKB')
  if ($left) {
    $lc = CountRec (Get-NS $left)
    Log ("Documents\MyKB exists, files = " + $lc + " -> attempting delete")
    $done = $false
    foreach ($v in @($left.Verbs())) {
      if ($v.Name -match 'delete|删除') { try { $v.DoIt(); $done = $true } catch {} ; break }
    }
    if (-not $done) { try { $left.InvokeVerb('delete'); $done = $true } catch {} }
    Start-Sleep -Seconds 8
    if ($docNS.ParseName('MyKB')) { Log "DELETE_RESULT: still exists (need manual delete)" }
    else { Log "DELETE_RESULT: removed" }
  } else { Log "Documents\MyKB does not exist (nothing to clean)" }
} catch {
  Log ("ERROR: " + $_.Exception.ToString())
}
$out | Out-File -FilePath 'C:\Code\MyKB\.sync-tmp2\cleanup_result.txt' -Encoding utf8
