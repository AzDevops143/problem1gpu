Write-Host "Fetching and syncing with GitHub (origin/main)..." -ForegroundColor Cyan

git fetch origin main

git pull --rebase origin main

git add -A

$status = git status --porcelain
if ($status) {
    $commitMsg = "Sync update: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Host "Committing local changes: $commitMsg" -ForegroundColor Yellow
    git commit -m "$commitMsg"
} else {
    Write-Host "No uncommitted local changes." -ForegroundColor Green
}

Write-Host "Pushing commits to GitHub..." -ForegroundColor Cyan
git push origin main

Write-Host "Repository is fully in sync!" -ForegroundColor Green
