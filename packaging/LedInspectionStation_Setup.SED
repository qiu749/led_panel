[Version]
Class=IEXPRESS
SEDVersion=3

[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=0
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=N
InstallPrompt=
DisplayLicense=
FinishMessage=
TargetName=D:\led-visual-inspection\release\LedInspectionStation_Setup.exe
FriendlyName=LedInspectionStation Setup
AppLaunched=powershell.exe -NoProfile -ExecutionPolicy Bypass -File install_station.ps1
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
SourceFiles=SourceFiles

[SourceFiles]
SourceFiles0=D:\led-visual-inspection\release\
SourceFiles1=D:\led-visual-inspection\packaging\

[SourceFiles0]
LedInspectionStation_green_selfcontained.zip=

[SourceFiles1]
install_station.ps1=
