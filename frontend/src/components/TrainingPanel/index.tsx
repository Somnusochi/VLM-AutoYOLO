import { Select, InputNumber, Dropdown } from "antd";
import { useAppStore } from "@/store/useAppStore";
import { JobHistoryList } from "@/components/Training/JobHistoryList";

// ── Component ───────────────────────────────────────

interface Props {
  detections: Detection[];
}

interface Props {
  detections: Detection[];
  total: number;
  hasNextPage: boolean;
  isFetchingNextPage: boolean;
  fetchNextPage: () => void;
}

export function TrainingPanel({ detections, total, hasNextPage, isFetchingNextPage, fetchNextPage }: Props) {
  const { t } = useTranslation();
  const isTraining = useAppStore((s) => s.isTraining);
  const setIsTraining = useAppStore((s) => s.setIsTraining);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [importModalOpen, setImportModalOpen] = useState(false);
  const [series, setSeries] = useState("yolo26");
  const [variant, setVariant] = useState("yolo26n");
  const [epochs, setEpochs] = useState(DEFAULT_EPOCHS);
  const [imgsz, setImgsz] = useState(DEFAULT_IMGSZ);
  const [batch, setBatch] = useState(DEFAULT_BATCH);
  const [splitPreset, setSplitPreset] = useState("70/20/10");
  const [taskType, setTaskType] = useState("detect");

  const splitPresets: Record<string, { train: number; val: number }> = {
    "70 / 20 / 10": { train: 0.7, val: 0.2 },
    "80 / 20": { train: 0.8, val: 0.2 },
    "90 / 10": { train: 0.9, val: 0.1 },
    "60 / 20 / 20": { train: 0.6, val: 0.2 },
  };
  const qc = useQueryClient();

  const variantsQuery = useQuery({
    queryKey: ["yolo-variants"],
    queryFn: fetchYoloSeries,
    staleTime: Infinity,
  });

  const seriesOptions = variantsQuery.data ?? {};
  const variantOptions = seriesOptions[series]?.variants ?? {};
  const variantKeys = Object.keys(variantOptions);
  const currentVariant = variant in variantOptions ? variant : (variantKeys[0] ?? variant);

  const jobsQuery = useInfiniteQuery({
    queryKey: ["training-jobs"],
    queryFn: ({ pageParam }) => fetchTrainingJobs(pageParam, 30),
    getNextPageParam: (lastPage, allPages) => {
      const totalFetched = allPages.reduce((sum, p) => sum + p.items.length, 0);
      return totalFetched < lastPage.total ? allPages.length + 1 : undefined;
    },
    initialPageParam: 1,
    refetchInterval: (query) => {
      const allItems = query.state.data?.pages.flatMap((p) => p.items) ?? [];
      if (allItems.length === 0) return false;
      const hasActiveJobs = allItems.some((j) => j.status === "running" || j.status === "pending");
      return hasActiveJobs ? 10_000 : false;
    },
    staleTime: 5_000,
  });

  const allJobs = useMemo(
    () => jobsQuery.data?.pages.flatMap((p) => p.items) ?? [],
    [jobsQuery.data],
  );

  useEffect(() => {
    setIsTraining(allJobs.some((j) => j.status === "running" || j.status === "pending"));
  }, [allJobs, setIsTraining]);

  const trainMut = useMutation({
    mutationFn: startTraining,
    onSuccess: () => {
      toast.success(t("trainingPanel.trainStarted"));
      qc.invalidateQueries({ queryKey: ["training-jobs"] });
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const selectAll = () => {
    const targets = trainFilter.size > 0 ? filteredDetections : detections;
    if (selected.size === targets.length && targets.every((d) => selected.has(d.id))) {
      setSelected(new Set());
    } else {
      setSelected(new Set(targets.map((d) => d.id)));
    }
  };

  const handleTrain = () => {
    if (selected.size === 0) {
      toast.error(t("trainingPanel.selectRecordRequired"));
      return;
    }
    setIsTraining(true);
    const p = splitPresets[splitPreset] ?? { train: 0.7, val: 0.2 };
    trainMut.mutate({
      detectionIds: [...selected],
      modelVariant: currentVariant,
      epochs,
      imgsz,
      batch,
      trainRatio: p.train,
      valRatio: p.val,
      taskType,
    });
  };

  const handleDownloadDataset = async (format: string, label: string) => {
    if (selectedCount === 0) {
      toast.error(t("trainingPanel.selectRecordRequired"));
      return;
    }
    try {
      const blob = await exportBatch([...selected], format);
      downloadBlob(blob, `${label}_dataset.zip`);
      toast.success(t("trainingPanel.datasetDownloaded"));
    } catch {
      toast.error(t("trainingPanel.datasetDownloadFailed"));
    }
  };

  const handleDownloadAllYoloSeg = async () => {
    try {
      const blob = await exportAll("yolo-seg");
      downloadBlob(blob, "YOLO_Seg_all_dataset.zip");
      toast.success(t("trainingPanel.datasetDownloaded"));
    } catch {
      toast.error(t("trainingPanel.datasetDownloadFailed"));
    }
  };

  // Tag filter for training candidates
  const trainCategories = useMemo(() => {
    const count = new Map<string, number>();
    detections.forEach((d) => {
      parseCategories(d.categories).forEach((c) => count.set(c, (count.get(c) ?? 0) + 1));
    });
    return [...count.entries()].sort((a, b) => b[1] - a[1]);
  }, [detections]);
  const [trainFilter, setTrainFilter] = useState<Set<string>>(new Set());
  const filteredDetections =
    trainFilter.size === 0
      ? detections
      : detections.filter((d) => parseCategories(d.categories).some((c) => trainFilter.has(c)));

  const selectedCount = selected.size;

  const [loadingAll, handleLoadAll] = useLoadAll(fetchNextPage);

  const [hoveredDetId, setHoveredDetId] = useState<string | null>(null);
  const [hoveredRect, setHoveredRect] = useState<{ right: number; top: number } | null>(null);
  const leaveTimerRef = useRef<number | null>(null);
  const enterTimerRef = useRef<number | null>(null);

  return (
    <div className="space-y-3">
      {/* Tag filter */}
      {trainCategories.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {trainCategories.map(([name, count]) => (
            <button
              key={name}
              type="button"
              onClick={() =>
                setTrainFilter((prev) => {
                  const next = new Set(prev);
                  if (next.has(name)) {
                    next.delete(name);
                  } else {
                    next.add(name);
                  }
                  return next;
                })
              }
              className={`rounded px-1 py-[2px] text-[11px] font-medium border transition-colors ${
                trainFilter.has(name)
                  ? "border-green-600 bg-green-600/20 text-green-500"
                  : "border-gray-200 bg-transparent text-gray-400 hover:border-gray-300 hover:text-gray-600"
              }`}
            >
              {name} ({count})
            </button>
          ))}
        </div>
      )}

      {/* Selection summary */}
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-500">
          {t("trainingPanel.selectedCountTotal", {
            current: selectedCount,
            filtered: trainFilter.size > 0 ? ` / ${filteredDetections.length}` : "",
            total: detections.length,
          })}
          {hasNextPage && (
            <button
              type="button"
              disabled={loadingAll || isFetchingNextPage}
              onClick={handleLoadAll}
              className="ml-2 text-primary-600 hover:text-primary-700 disabled:opacity-50"
            >
              {loadingAll
                ? t("common.loading")
                : t("trainingPanel.loadAll", { remaining: total - detections.length })}
            </button>
          )}
        </span>
        <div className="flex items-center gap-2">
          {total > 0 && (
            <button
              type="button"
              onClick={handleDownloadAllYoloSeg}
              className="text-xs text-primary-600 hover:underline"
            >
              {t("trainingPanel.downloadAllYoloSeg")}
            </button>
          )}
          {selectedCount > 0 && (
            <Dropdown
              menu={{
                items: [
                  { key: "yolo", label: "YOLO (.txt)" },
                  { key: "yolo-seg", label: "YOLO Segmentation" },
                  { key: "coco", label: "COCO (.json)" },
                  { key: "voc", label: "Pascal VOC (.xml)" },
                  { key: "createml", label: "CreateML (.json)" },
                ],
                onClick: ({ key }) => {
                  const labels: Record<string, string> = {
                    yolo: "YOLO",
                    "yolo-seg": "YOLO_Seg",
                    coco: "COCO",
                    voc: "VOC",
                    createml: "CreateML",
                  };
                  handleDownloadDataset(key, labels[key] ?? key);
                },
              }}
              trigger={["click"]}
            >
              <button type="button" className="text-xs text-primary-600 hover:underline">
                {t("trainingPanel.downloadDataset")}
              </button>
            </Dropdown>
          )}
          <button
            type="button"
            onClick={() => setImportModalOpen(true)}
            className="text-xs text-primary-600 hover:underline"
          >
            {t("trainingPanel.importDataset")}
          </button>
          <button
            type="button"
            onClick={selectAll}
            className="text-xs text-primary-600 hover:underline"
          >
            {(() => {
              const targets = trainFilter.size > 0 ? filteredDetections : detections;
              return selectedCount === targets.length && targets.every((d) => selected.has(d.id))
                ? t("trainingPanel.deselectAll")
                : t("trainingPanel.selectAll");
            })()}
          </button>
        </div>
      </div>

      {/* Detection checklist with thumbnails */}
      <TrainingCandidateList
        filteredDetections={filteredDetections}
        selected={selected}
        toggleSelect={toggleSelect}
        setHoveredDetId={setHoveredDetId}
        setHoveredRect={setHoveredRect}
        enterTimerRef={enterTimerRef}
        leaveTimerRef={leaveTimerRef}
        hasNextPage={hasNextPage}
        isFetchingNextPage={isFetchingNextPage}
        fetchNextPage={fetchNextPage}
      />

      {/* Global single-instance hover preview popup */}
      {hoveredDetId && hoveredRect && (
        <HoverPreview
          detId={hoveredDetId}
          hoveredRect={hoveredRect}
          setHoveredDetId={setHoveredDetId}
          setHoveredRect={setHoveredRect}
          leaveTimerRef={leaveTimerRef}
        />
      )}

      {/* Training params */}
      <div className="grid grid-cols-2 gap-3 training-params">
        <div>
          <label className="text-xs text-gray-500 mb-1 block">{t("trainingPanel.series")}</label>
          <Select
            value={series}
            onChange={setSeries}
            options={Object.entries(seriesOptions).map(([key, s]) => ({
              value: key,
              label: s.label,
            }))}
            className="w-full"
            showSearch
            filterOption={(input, option) =>
              (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
            }
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 mb-1 block">
            {t("trainingPanel.specification")}
          </label>
          <Select
            value={currentVariant}
            onChange={setVariant}
            options={Object.entries(variantOptions).map(([key, label]) => ({ value: key, label }))}
            className="w-full"
            showSearch
            filterOption={(input, option) =>
              (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
            }
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 mb-1 block">{t("trainingPanel.epochs")}</label>
          <InputNumber
            min={1}
            max={9999}
            value={epochs}
            onChange={(v) => setEpochs(v ?? DEFAULT_EPOCHS)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 mb-1 block">{t("trainingPanel.imgsz")}</label>
          <InputNumber
            min={128}
            max={4096}
            step={64}
            value={imgsz}
            onChange={(v) => setImgsz(v ?? DEFAULT_IMGSZ)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 mb-1 block">{t("trainingPanel.batch")}</label>
          <InputNumber
            min={1}
            max={256}
            value={batch}
            onChange={(v) => setBatch(v ?? DEFAULT_BATCH)}
            className="w-full"
          />
        </div>
        <div>
          <label className="text-xs text-gray-500 mb-1 block">
            {t("trainingPanel.splitRatio", {
              test: splitPreset.split("/").length >= 3 ? t("trainingPanel.splitRatioTest") : "",
            })}
          </label>
          <Select
            value={splitPreset}
            onChange={setSplitPreset}
            options={Object.keys(splitPresets).map((k) => ({ value: k, label: k }))}
            className="w-full"
          />
        </div>
        <div className="col-span-2">
          <label className="text-xs text-gray-500 mb-1 block">{t("trainingPanel.taskType")}</label>
          <Select
            value={taskType}
            onChange={setTaskType}
            options={[
              { value: "detect", label: t("trainingPanel.taskDetect") },
              { value: "segment", label: t("trainingPanel.taskSegment") },
              { value: "classify", label: t("trainingPanel.taskClassify"), disabled: true },
              { value: "pose", label: t("trainingPanel.taskPose"), disabled: true },
              { value: "obb", label: t("trainingPanel.taskObb"), disabled: true },
            ]}
            className="w-full"
          />
        </div>
      </div>

      {/* Train button */}
      <button
        type="button"
        disabled={trainMut.isPending || selectedCount === 0}
        onClick={handleTrain}
        className="w-full rounded bg-green-600 py-2 text-sm font-semibold text-white hover:bg-green-700 disabled:opacity-50 transition-colors"
      >
        {trainMut.isPending
          ? t("trainingPanel.starting")
          : isTraining
            ? t("trainingPanel.addToQueue", { count: selectedCount })
            : t("trainingPanel.startTrainCount", { count: selectedCount })}
      </button>

      {/* Job history */}
      {allJobs.length > 0 && (
        <JobHistoryList
          jobs={allJobs}
          hasNextPage={jobsQuery.hasNextPage ?? false}
          isFetchingNextPage={jobsQuery.isFetchingNextPage}
          fetchNextPage={() => jobsQuery.fetchNextPage()}
        />
      )}
      <DatasetImportModal
        open={importModalOpen}
        onClose={() => setImportModalOpen(false)}
      />
    </div>
  );
}
