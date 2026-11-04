# 同步到手机.ps1 —— 将本知识库增量同步到手机 Obsidian/MyKB
# 用法: 插上手机(文件传输模式)后, 双击 同步到手机.bat, 或在此脚本后加 -NoPause 参数运行
param([switch]$NoPause)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 增量判断依据: 文件大小 (MTP 上报的修改时间不可靠, 故不用时间比较)
# 已知限制: 内容改了但字节数恰好不变的文件不会被识别(极少见)

$SkipNames = @('.git', 'nul', 'Thumbs.db', '.DS_Store', 'workspace.json', 'workspace-mobile.json')  # 任意层级跳过这些名字
$SkipExt   = @('.log')                                          # 跳过后缀

$Vault   = Split-Path $PSScriptRoot -Parent     # 知识库根目录 = 脚本的上一级
$ToolDir = $PSScriptRoot
$LogFile = Join-Path $ToolDir 'sync.log'
$LogLines = New-Object System.Collections.ArrayList

function Log($m) { [void]$LogLines.Add($m); Write-Host $m }
function Log-Red($m) { [void]$LogLines.Add($m); Write-Host $m -ForegroundColor Red }
function Log-Green($m) { [void]$LogLines.Add($m); Write-Host $m -ForegroundColor Green }

# ===== 静默删除 MTP 文件的 COM 互操作定义 (IFileOperation + FOF_NOCONFIRMATION, 不弹窗) =====
$csharpCode = @'
using System;
using System.Runtime.InteropServices;

public static class MtpSilentDelete
{
    [ComImport, Guid("3ad05575-8857-4850-9277-11b85bdb8e09")]
    private class FileOperation { }

    [ComImport, Guid("947aab5f-0a5c-4c13-b4d6-4bf7836fc9f8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    private interface IFileOperation
    {
        void Advise([In] IntPtr pfops, [Out] out uint pdwCookie);
        void Unadvise([In] uint dwCookie);
        void SetOperationFlags([In] uint dwOperationFlags);
        void SetProgressMessage([In, MarshalAs(UnmanagedType.LPWStr)] string pszMessage);
        void SetProgressDialog([In] IntPtr pops);
        void SetProperties([In] IntPtr pproparray);
        void SetOwnerWindow([In] IntPtr hwndOwner);
        void ApplyPropertiesToItem([In] IntPtr psiItem);
        void ApplyPropertiesToItems([In] IntPtr punkItems);
        void RenameItem([In] IntPtr psiItem, [In, MarshalAs(UnmanagedType.LPWStr)] string pszNewName);
        void RenameItems([In] IntPtr punkItems, [In, MarshalAs(UnmanagedType.LPWStr)] string pszNewName);
        void MoveItem([In] IntPtr psiItem, [In] IntPtr psiDestinationFolder, [In, MarshalAs(UnmanagedType.LPWStr)] string pszNewName);
        void MoveItems([In] IntPtr punkItems, [In] IntPtr psiDestinationFolder);
        void CopyItem([In] IntPtr psiItem, [In] IntPtr psiDestinationFolder, [In, MarshalAs(UnmanagedType.LPWStr)] string pszCopyName);
        void CopyItems([In] IntPtr punkItems, [In] IntPtr psiDestinationFolder);
        void DeleteItem([In] IntPtr psiItem, [In] IntPtr pfopsItem);
        void DeleteItems([In] IntPtr punkItems);
        void NewItem([In] IntPtr psiDestinationFolder, [In] uint dwFileAttributes, [In, MarshalAs(UnmanagedType.LPWStr)] string pszName, [In, MarshalAs(UnmanagedType.LPWStr)] string pszTemplateName);
        void PerformOperations();
        [return: MarshalAs(UnmanagedType.Bool)] bool GetAnyOperationsAborted();
    }

    [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
    private static extern int SHCreateItemFromParsingName(
        [In, MarshalAs(UnmanagedType.LPWStr)] string pszPath,
        [In] IntPtr pbc,
        [In] ref Guid riid,
        [Out] out IntPtr ppv);

    public static int DeleteSilent(string path)
    {
        IntPtr item = IntPtr.Zero;
        try
        {
            Guid iidShellItem = new Guid("b63ea76d-1f85-456f-a19c-48159efa858b");
            int hr = SHCreateItemFromParsingName(path, IntPtr.Zero, ref iidShellItem, out item);
            if (hr != 0) return hr;
            Type t = Type.GetTypeFromCLSID(new Guid("3ad05575-8857-4850-9277-11b85bdb8e09"));
            if (t == null) return -1;
            IFileOperation fo = (IFileOperation)Activator.CreateInstance(t);
            const uint FOF_SILENT = 0x0004;
            const uint FOF_NOCONFIRMATION = 0x0010;
            const uint FOF_NOERRORUI = 0x0400;
            fo.SetOperationFlags(FOF_SILENT | FOF_NOCONFIRMATION | FOF_NOERRORUI);
            fo.DeleteItem(item, IntPtr.Zero);
            fo.PerformOperations();
            return 0;
        }
        catch (Exception ex)
        {
            return Marshal.GetHRForException(ex);
        }
        finally
        {
            if (item != IntPtr.Zero) Marshal.Release(item);
        }
    }
}
'@

$SilentDeleteAvailable = $false
try {
  Add-Type -TypeDefinition $csharpCode -ErrorAction Stop
  $SilentDeleteAvailable = $true
} catch {
  Write-Host '静默删除组件编译失败, 将使用传统删除方式(可能弹窗)' -ForegroundColor Yellow
}

function Get-NS([object]$item) {
  $f = $item.GetFolder()
  if ($f) { return $f }
  return $shell.Namespace($item.Path)
}

function Test-Skip([string]$name, [string]$full) {
  if ($SkipNames -contains $name) { return $true }
  if ($name -like '.sync-tmp*') { return $true }   # 临时目录(.sync-tmp/.sync-tmp2 等)一律跳过
  foreach ($e in $SkipExt) { if ($full.ToLower().EndsWith($e)) { return $true } }
  return $false
}

# MTP 设备上 FolderItem.Size 返回 0, 必须用扩展属性取真实大小
function Get-FileSize($item) {
  try {
    $v = $item.ExtendedProperty('System.Size')
    if ($v) { return [long]$v }
  } catch { }
  try { if ($item.Size) { return [long]$item.Size } } catch { }
  return $null
}

# 删除手机端文件 (先删后拷, 避免 MTP 覆盖文件不可靠)
function Delete-ItemMTP($dstNS, $name) {
  $it = $dstNS.ParseName($name)
  if (-not $it) { return $true }
  if ($SilentDeleteAvailable) {
    $hr = [MtpSilentDelete]::DeleteSilent($it.Path)
    if ($hr -eq 0) {
      $deadline = (Get-Date).AddSeconds(30)
      while ((Get-Date) -lt $deadline) {
        Start-Sleep -Milliseconds 800
        if ($null -eq $dstNS.ParseName($name)) { return $true }
      }
      return $false
    }
  }
  # 兜底: 传统方式(极少触发, 可能弹窗)
  $done = $false
  foreach ($v in @($it.Verbs())) {
    if ($v.Name -match 'delete|删除') { try { $v.DoIt(); $done = $true } catch { }; break }
  }
  if (-not $done) { try { $it.InvokeVerb('delete') } catch { } }
  $deadline = (Get-Date).AddSeconds(30)
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 800
    if ($null -eq $dstNS.ParseName($name)) { return $true }
  }
  return $false
}

# 复制单个文件并等待落盘(校验大小一致)
function Copy-And-Wait($dstNS, $srcItem) {
  $dstNS.CopyHere($srcItem.Path, 84)
  $deadline = (Get-Date).AddSeconds(60)
  while ((Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 800
    $d = $dstNS.ParseName($srcItem.Name)
    if ($d) {
      try {
        $ds = Get-FileSize $d; $ss = Get-FileSize $srcItem
        if ($null -eq $ds -or $ds -eq $ss) { return $true }
      } catch { return $true }
    }
  }
  return $false
}

function Sync-File($dstNS, $srcItem) {
  [void](Delete-ItemMTP $dstNS $srcItem.Name)
  return Copy-And-Wait $dstNS $srcItem
}

function WalkSync($srcNS, $dstNS, $rel, $stats) {
  foreach ($it in @($srcNS.Items())) {
    if (Test-Skip $it.Name ($rel + '/' + $it.Name)) { continue }
    $dstItem = $dstNS.ParseName($it.Name)
    if ($it.IsFolder) {
      if (-not $dstItem) {
        $dstNS.NewFolder($it.Name)
        $dstItem = $dstNS.ParseName($it.Name)
      }
      if (-not $dstItem) { throw ('无法在手机创建文件夹: ' + $rel + '/' + $it.Name) }
      WalkSync (Get-NS $it) (Get-NS $dstItem) ($rel + '/' + $it.Name) $stats
    } else {
      $need = $true
      if ($dstItem) {
        $need = $false
        try {
          $ss = Get-FileSize $it; $ds = Get-FileSize $dstItem
          if ($null -eq $ss -or $null -eq $ds -or $ds -ne $ss) { $need = $true }
        } catch { $need = $true }
      }
      if ($need) {
        Log ("更新: " + $rel + '/' + $it.Name)
        if (Sync-File $dstNS $it) { $stats.Copied++ }
        else { Log-Red ("更新失败(超时): " + $rel + '/' + $it.Name); $stats.Failed++ }
      } else {
        $stats.Same++
      }
    }
  }
}

function Verify($srcNS, $dstNS, $rel, $bad, $stats) {
  foreach ($it in @($srcNS.Items())) {
    if (Test-Skip $it.Name ($rel + '/' + $it.Name)) { continue }
    if ($it.IsFolder) {
      $d = $dstNS.ParseName($it.Name)
      if (-not $d) { [void]$bad.Add($rel + '/' + $it.Name + ' (文件夹缺失)'); continue }
      Verify (Get-NS $it) (Get-NS $d) ($rel + '/' + $it.Name) $bad $stats
    } else {
      $stats.Total++
      $d = $dstNS.ParseName($it.Name)
      if (-not $d) { [void]$bad.Add($rel + '/' + $it.Name + ' (缺失)'); continue }
      try { $ss = Get-FileSize $it; $ds = Get-FileSize $d; if ($ss -ne $ds) { [void]$bad.Add($rel + '/' + $it.Name + ' (大小不符)') } } catch { }
    }
  }
}

function Find-Phone {
  # 优先: WPD 便携设备类 GUID 特征
  foreach ($it in @($shell.Namespace(17).Items())) {
    if ($it.Path -match '6ac27878-a6fa-4155-ba85-f98f491d4f33') { return $it }
  }
  # 回退: 非盘符、非云盘的文件夹
  foreach ($it in @($shell.Namespace(17).Items())) {
    if ($it.IsFolder -and $it.Name -notmatch 'WPS' -and $it.Name -notmatch '\(') { return $it }
  }
  return $null
}

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
try {
  Log ("知识库: " + $Vault)
  $shell = New-Object -ComObject Shell.Application

  $dev = Find-Phone
  if (-not $dev) {
    Log-Red '未检测到手机。请确认: ① 数据线已连接; ② 手机通知栏把 USB 模式选为「文件传输 (MTP)」; ③ 弹出授权时点允许。'
    throw 'device not found'
  }
  Log ("设备: " + $dev.Name)

  $devNS = Get-NS $dev
  $storage = @($devNS.Items())[0]
  if (-not $storage) { throw '手机存储不可用, 请确认 USB 模式为「文件传输 (MTP)」' }
  $storageNS = Get-NS $storage

  $obs = $storageNS.ParseName('Obsidian')
  if (-not $obs) { $storageNS.NewFolder('Obsidian'); $obs = $storageNS.ParseName('Obsidian') }
  if (-not $obs) { throw '无法创建手机端 Obsidian 文件夹' }
  $obsNS = Get-NS $obs
  $mkb = $obsNS.ParseName('MyKB')
  if (-not $mkb) { $obsNS.NewFolder('MyKB'); $mkb = $obsNS.ParseName('MyKB') }
  if (-not $mkb) { throw '无法创建手机端 MyKB 文件夹' }
  $dstNS = Get-NS $mkb
  Log ("目标: " + $dev.Name + " -> Obsidian/MyKB")

  $srcNS = $shell.Namespace($Vault)
  $stats = @{ Copied = 0; Same = 0; Failed = 0; Total = 0 }

  Log '--- 开始增量同步 ---'
  WalkSync $srcNS $dstNS '' $stats
  Log ("本次: 更新 " + $stats.Copied + " 个, 未变化 " + $stats.Same + " 个, 失败 " + $stats.Failed + " 个")

  Log '--- 校验 ---'
  $bad = New-Object System.Collections.ArrayList
  Verify $srcNS $dstNS '' $bad $stats
  if ($bad.Count -eq 0 -and $stats.Failed -eq 0) {
    Log-Green ("校验通过: 电脑端 " + $stats.Total + " 个文件在手机端全部存在且大小一致")
  } else {
    Log-Red ("校验发现问题: " + ($bad -join '; ') + ($stats.Failed) + " 个更新失败")
  }
} catch {
  Log-Red ("错误: " + $_.Exception.Message)
} finally {
  $stopwatch.Stop()
  Log ("用时 " + [Math]::Round($stopwatch.Elapsed.TotalSeconds, 1) + " 秒")
  $LogLines | Out-File -FilePath $LogFile -Encoding utf8
  if (-not $NoPause) {
    Write-Host ''
    Read-Host '按回车键关闭窗口'
  }
}
