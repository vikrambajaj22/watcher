export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="p-4 glass border-danger/40 rounded-xl mb-4">
      <strong className="text-danger">Error: </strong>
      {message}
    </div>
  );
}
